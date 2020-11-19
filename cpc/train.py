# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import json
import os
import numpy as np
import torch
import time
from copy import deepcopy
import random
import psutil
import sys
import torchaudio

import cpc.criterion as cr
import cpc.model as model
import cpc.utils.misc as utils
import cpc.feature_loader as fl
from cpc.balance_sampler import get_balance_sampler
from cpc.cpc_default_config import set_default_cpc_config
from cpc.dataset import AudioBatchData, findAllSeqs, filterSeqs, parseSeqLabels, \
                        PeakNorm
from cpc.criterion.research import CPCBertCriterion, DeepEmbeddedClustering, \
    DeepClustering, CTCCLustering, buildNewPhoneDict
from cpc.distributed_training.distributed_mode import init_distributed_mode
from cpc.data_augmentation import augmentation_factory


def getCriterion(args, downsampling, nSpeakers, nPhones):
    dimFeatures = args.hiddenGar if not args.onEncoder else args.hiddenEncoder
    if not args.supervised:
        if args.cpc_mode == "bert":
            cpcCriterion = CPCBertCriterion(args.hiddenGar,
                                            args.hiddenEncoder,
                                            args.negativeSamplingExt)
        elif args.cpc_mode == 'none':
            cpcCriterion = cr.NoneCriterion()
            args.cluster_delay = 0
        else:
            mode = "cumNorm" if args.normMode == "cumNorm" else args.cpc_mode
            sizeInputSeq = (args.sizeWindow // downsampling)
            cpcCriterion = cr.CPCUnsupersivedCriterion(args.nPredicts,
                                                       args.hiddenGar,
                                                       args.hiddenEncoder,
                                                       args.negativeSamplingExt,
                                                       mode=mode,
                                                       rnnMode=args.rnnMode,
                                                       dropout=args.dropout,
                                                       nSpeakers=nSpeakers,
                                                       speakerEmbedding=args.speakerEmbedding,
                                                       sizeInputSeq=sizeInputSeq,
                                                       multihead_rnn=args.multihead_rnn,
                                                       transformer_pruning=args.transformer_pruning)
    elif args.pathPhone is not None:
        if not args.CTC:
            cpcCriterion = cr.PhoneCriterion(dimFeatures,
                                             nPhones, args.onEncoder,
                                             nLayers=args.nLevelsPhone)
        else:
            cpcCriterion = cr.CTCPhoneCriterion(dimFeatures,
                                                nPhones, args.onEncoder)
    else:
        cpcCriterion = cr.SpeakerCriterion(dimFeatures, nSpeakers)
    return cpcCriterion


def loadCriterion(pathCheckpoint, downsampling, nSpeakers, nPhones):
    _, _, locArgs = fl.getCheckpointData(os.path.dirname(pathCheckpoint))
    criterion = getCriterion(locArgs, downsampling, nSpeakers, nPhones)

    state_dict = torch.load(pathCheckpoint, 'cpu')

    criterion.load_state_dict(state_dict["cpcCriterion"])
    return criterion


def adversarialTrainStep(dataLoader, cpcModel,
                         cpcCriterion, optimizerCPC,
                         speakerCriterion, optimizerPhone,
                         clustering, loggingStep):

    cpcModel.train()
    speakerCriterion.train()
    cpcCriterion.train()
    start_time = time.perf_counter()

    logs = {"loss_train_speak": 0, "acc_train_speak": 0}
    iter, lastlogs, n_examples = 0, None, 0
    for step, full_data in enumerate(dataLoader):

        optimizerCPC.zero_grad()
        optimizerPhone.zero_grad()

        sequence, label = [x.cuda(non_blocking=True) for x in full_data]
        past, future = sequence[:, 0, ...], sequence[:, 1, ...]

        b = past.size(0)
        n_examples += b

        combined = torch.cat([past, future], dim=0)
        label = torch.cat([label, label])

        c_feature, encoded_data, label = cpcModel(combined, label)
        c_feature = c_feature[:b, :, :]
        encoded_data = encoded_data[b:, :, :]
        label =label[:b]

        allLosses, allAcc = cpcCriterion(c_feature, encoded_data, label)
        lossSpeak, _ = speakerCriterion(c_feature, encoded_data, None)
        totLoss = allLosses.sum() + lossSpeak.sum()

        if clustering is not None:
            lossCluster = clustering(c_feature, labelPhone)
            totLoss += lossCluster.sum()

        if "locLoss_train_cpc" not in logs:
            logs["locLoss_train_cpc"] = np.zeros(allLosses.size(1))
            logs["locAcc_train_cpc"] = np.zeros(allLosses.size(1))
            if clustering is not None:
                logs["lossCluster_train"] = np.zeros(lossCluster.size(1))

        logs["loss_train_speak"] += (lossSpeak.mean(dim=0).view(1)
                                     ).detach().cpu().numpy()

        logs["locLoss_train_cpc"] += (allLosses.mean(dim=0)
                                      ).detach().cpu().numpy()
        if clustering is not None:
            logs["lossCluster_train"] += (lossCluster.mean(dim=0)
                                          ).detach().cpu().numpy()
        logs["locAcc_train_cpc"] += (allAcc.mean(dim=0)).cpu().numpy()

        if clustering is not None:
            totLoss += lossCluster.sum()
        totLoss.backward()
        optimizerCPC.step()
        optimizerPhone.zero_grad()

        lossSpeak, accSpeak = speakerCriterion(
            c_feature.detach(), encoded_data.detach(), label)

        totLoss = lossSpeak.sum()
        totLoss.backward()
        optimizerPhone.step()

        logs["acc_train_speak"] += (accSpeak.mean(dim=0)).cpu().numpy()
        iter += 1

        if (step + 1) % loggingStep == 0:
            new_time = time.perf_counter()
            elapsed = new_time - start_time
            print(f"Update {step + 1}")
            print(f"elapsed: {elapsed:.1f} s")
            print(
                f"{1000.0 * elapsed / loggingStep:.1f} ms per batch, {1000.0 * elapsed / n_examples:.1f} ms / example")
            locLogs = utils.update_logs(logs, loggingStep, lastlogs)
            lastlogs = deepcopy(logs)
            utils.show_logs("Training loss", locLogs)
            start_time, n_examples = new_time, 0

    logs = utils.update_logs(logs, iter)
    logs["iter"] = iter
    utils.show_logs(
        f"Average training loss on epoch ({iter+1} updates) :", logs)
    return logs


def trainStep(dataLoader,
              cpcModel,
              cpcCriterion,
              optimizer,
              scheduler,
              clustering,
              loggingStep):

    cpcModel.train()
    cpcCriterion.train()

    start_time = time.perf_counter()
    n_examples = 0
    logs, lastlogs = {}, None
    iter = 0
    for step, full_data in enumerate(dataLoader):
        sequence, label = [x.cuda(non_blocking=True) for x in full_data]
        past, future = sequence[:, 0, ...], sequence[:, 1, ...]

        b = past.size(0)
        n_examples += b

        combined = torch.cat([past, future], dim=0)
        label = torch.cat([label, label])

        c_feature, encoded_data, label = cpcModel(combined, label)
        c_feature = c_feature[:b, :, :]
        encoded_data = encoded_data[b:, :, :]
        label =label[:b]

        allLosses, allAcc = cpcCriterion(c_feature, encoded_data, label)
        totLoss = allLosses.sum()
        totLoss.backward()

        if clustering is not None:
            lossCluster = clustering(c_feature, label)
            totLoss += lossCluster.sum()

        # Show grads ?
        optimizer.step()
        optimizer.zero_grad()

        if allLosses.nelement() > 0:
            if "locLoss_train" not in logs:
                logs["locLoss_train"] = np.zeros(allLosses.size(1))
                logs["locAcc_train"] = np.zeros(allLosses.size(1))
                if clustering is not None:
                    logs["lossCluster_train"] = np.zeros(lossCluster.size(1))

            iter += 1
            logs["locLoss_train"] += (allLosses.mean(dim=0)).detach().cpu().numpy()
            logs["locAcc_train"] += (allAcc.mean(dim=0)).cpu().numpy()
            if clustering is not None:
                logs["lossCluster_train"] += (lossCluster.mean(dim=0)
                                              ).detach().cpu().numpy()
            if (step + 1) % loggingStep == 0:
                new_time = time.perf_counter()
                elapsed = new_time - start_time
                print(f"Update {step + 1}")
                print(f"elapsed: {elapsed:.1f} s")
                print(
                    f"{1000.0 * elapsed / loggingStep:.1f} ms per batch, {1000.0 * elapsed / n_examples:.1f} ms / example")
                locLogs = utils.update_logs(logs, loggingStep, lastlogs)
                lastlogs = deepcopy(logs)
                utils.show_logs("Training loss", locLogs)
                start_time, n_examples = new_time, 0

    if scheduler is not None:
        scheduler.step()

    logs = utils.update_logs(logs, iter)
    logs["iter"] = iter
    utils.show_logs("Average training loss on epoch", logs)
    return logs


def valStep(dataLoader,
            cpcModel,
            cpcCriterion,
            clustering):

    cpcCriterion.eval()
    cpcModel.eval()
    logs = {}
    cpcCriterion.eval()
    cpcModel.eval()
    iter = 0

    for step, full_data in enumerate(dataLoader):
        sequence, label = [x.cuda(non_blocking=True) for x in full_data]

        past, future = sequence[:, 0, ...], sequence[:, 1, ...]
        label = torch.cat([label, label])

        b = past.size(0)

        with torch.no_grad():
            combined = torch.cat([past, future], dim=0)
            c_feature, encoded_data, label = cpcModel(combined, label)
            c_feature = c_feature[:b, ...]
            encoded_data = encoded_data[b:, ...]
            label =label[:b]

            allLosses, allAcc = cpcCriterion(c_feature, encoded_data, label)
            if clustering is not None:
                lossCluster = clustering(c_feature, label)

        if "locLoss_val" not in logs:
            logs["locLoss_val"] = np.zeros(allLosses.size(1))
            logs["locAcc_val"] = np.zeros(allLosses.size(1))
            if clustering is not None:
                logs["lossCluster_val"] = np.zeros(lossCluster.size(1))

        iter += 1
        logs["locLoss_val"] += allLosses.mean(dim=0).cpu().numpy()
        if clustering is not None:
            logs["lossCluster_val"] += (lossCluster.mean(dim=0)
                                        ).detach().cpu().numpy()
        logs["locAcc_val"] += allAcc.mean(dim=0).cpu().numpy()

    logs = utils.update_logs(logs, iter)
    logs["iter"] = iter
    utils.show_logs("Validation loss:", logs)
    return logs


def run(trainDataset,
        valDataset,
        batchSize,
        samplingMode,
        balance_sampler,
        cpcModel,
        cpcCriterion,
        nEpoch,
        pathCheckpoint,
        optimizer,
        scheduler,
        logs,
        adversarial,
        clustering):

    print(f"Running {nEpoch} epochs")
    startEpoch = len(logs["epoch"])
    bestAcc = 0
    bestStateDict = None
    start_time = time.time()

    if adversarial is not None:
        optimAdv = torch.optim.Adam(list(adversarial.parameters()), lr=2e-4)

    for epoch in range(startEpoch, nEpoch):

        print(f"Starting epoch {epoch}")
        utils.cpu_stats()

        if clustering is not None:
            cpcModel.eval()
            trainDataset.doubleLabels = False
            clustering.module.updateCLusters(trainDataset.getDataLoader(batchSize, 'uniform',
                                                                        True, numWorkers=0),
                                             fl.FeatureModule(cpcModel.module, False))
            if clustering.module.canRun():
                for dataset, status in [(trainDataset, 'train'), (valDataset, 'val')]:
                    phoneLabels, phoneFill = \
                        buildNewPhoneDict(dataset.dbPath,
                                          dataset.getSeqNames(),
                                          cpcModel.module,
                                          clustering.module.clusters,
                                          clustering.module.k)
                    dataset.resetPhoneLabels(phoneLabels, 160)
                    fillingStatus = (phoneFill == 0).sum().item()
                    print(
                        f"{fillingStatus} clusters empty out of {clustering.module.k}")

        trainLoader = trainDataset.getDataLoader(batchSize, samplingMode,
                                                 True, numWorkers=0,
                                                 balance_sampler=balance_sampler)

        valLoader = valDataset.getDataLoader(batchSize, 'sequential', False,
                                             numWorkers=0) if valDataset else []

        print("Training dataset %d batches, Validation dataset %d batches, batch size %d" %
              (len(trainLoader), len(valLoader), batchSize))

        if adversarial is not None:
            locLogsTrain = adversarialTrainStep(trainLoader, cpcModel,
                                                cpcCriterion,
                                                optimizer, adversarial,
                                                optimAdv, clustering,
                                                logs["logging_step"])
        else:
            locLogsTrain = trainStep(
                trainLoader, cpcModel, cpcCriterion, optimizer,
                scheduler, clustering, logs["logging_step"])

        if valDataset:
            locLogsVal = valStep(valLoader, cpcModel, cpcCriterion, clustering)

        print(f'Ran {epoch + 1} epochs '
              f'in {time.time() - start_time:.2f} seconds')

        if valDataset:
            currentAccuracy = float(locLogsVal["locAcc_val"].mean())
            if currentAccuracy > bestAcc:
                bestStateDict = fl.get_module(cpcModel).state_dict()
        else:
            bestStateDict = fl.get_module(cpcModel).state_dict()

        items = dict(locLogsTrain, **locLogsVal).items() if valDataset else locLogsTrain.items()

        for key, value in items:
            if key not in logs:
                logs[key] = [None for x in range(epoch)]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            logs[key].append(value)

        logs["epoch"].append(epoch)

        if pathCheckpoint is not None \
                and (epoch % logs["saveStep"] == 0 or epoch == nEpoch-1):

            modelStateDict = fl.get_module(cpcModel).state_dict()
            criterionStateDict = fl.get_module(cpcCriterion).state_dict()

            fl.save_checkpoint(modelStateDict, criterionStateDict,
                               optimizer.state_dict(), bestStateDict,
                               f"{pathCheckpoint}_{epoch}.pt")
            utils.save_logs(logs, pathCheckpoint + "_logs.json")


def main(argv):
    args = parseArgs(argv)

    logs = {"epoch": [], "iter": [], "saveStep": args.save_step}
    logs["logging_step"] = args.logging_step
    load_optimizer = False

    if args.pathCheckpoint is not None and not args.restart:
        cdata = fl.getCheckpointData(args.pathCheckpoint)
        if cdata is not None:
            forbidden_attr = ["nGPU", "pathCheckpoint", "local_rank",
                              "global_rank","node_id", "n_gpu_per_node"
                              "debug", "restart", "world_size", "n_nodes"]
            data, logs, locArgs = cdata
            print(f"Checkpoint detected at {data}")
            default_values = { k : v for k, v in vars(locArgs).items() \
                           if k not in forbidden_attr}
            args = parseArgs(argv, default_values)
            args.load, load_optimizer = [data], True
            args.loadCriterion = True
    batchSize = args.nGPU * args.batchSizeGPU

    if args.distributed:
        print('Distributed mode, moving to 1 process for data loading')
        args.n_process_loader = 1
        init_distributed_mode(args)
    args.is_local_master = (not args.distributed) or (args.global_rank == 0)

    utils.set_seed(args.random_seed)

    print(f'CONFIG:\n{json.dumps(vars(args), indent=4, sort_keys=True)}')
    print('-' * 50)

    seqNames, speakers = findAllSeqs(args.pathDB,
                                     extension=args.file_extension,
                                     loadCache=not args.ignore_cache,
                                     cache_path=args.path_cache)

    print(f'Found files: {len(seqNames)} seqs, {len(speakers)} speakers')
    # Datasets
    if args.pathTrain is not None:
        seqTrain = filterSeqs(args.pathTrain, seqNames)
    else:
        seqTrain = seqNames

    if args.pathVal is None:
        print('No validation data specified!')
        seqVal = []
    else:
        seqVal = filterSeqs(args.pathVal, seqNames)

    if args.debug:
        seqTrain = seqTrain[-1000:]
        seqVal = seqVal[-100:]

    phoneLabels, nPhones = None, None
    if args.supervised and args.pathPhone is not None:
        print("Loading the phone labels at " + args.pathPhone)
        phoneLabels, nPhones = parseSeqLabels(args.pathPhone)
        print(f"{nPhones} phones found")

    # Noise data
    seqNoise = None
    noiseDataset = None

    if args.pathDBNoise is not None:
        seqNoise, _ = findAllSeqs(args.pathDBNoise,
                                   extension=args.noise_extension,
                                   loadCache=not args.ignore_cache,
                                   speaker_level=0)
        if args.pathSeqNoise is not None:
            seqNoise = filterSeqs(args.pathSeqNoise, seqNoise)
        if args.debug:
            seqNoise = seqNoise[:100]

        print("")
        print(f'Loading noise data at {args.pathDBNoise}')
        print("Loading the noise dataset")
        noiseDataset =  AudioBatchData(args.pathDBNoise, args.sizeWindow,
                                       seqNoise, None, 1,
                                       transform=PeakNorm(),
                                       nProcessLoader=args.n_process_loader,
                                       MAX_SIZE_LOADED=args.max_size_loaded)

    if args.distributed:
        import random
        random.Random(4).shuffle(seqTrain)
        def filter_distributed(files):
            start = len(files) * args.global_rank // args.world_size
            end = len(files) * (args.global_rank + 1) // args.world_size
            print(start, end)
            return files[start:end]
        print(
            f'Initial worker files: {len(seqTrain)} train, {len(seqVal)} val')
        seqTrain = filter_distributed(seqTrain)
        seqVal = filter_distributed(seqVal)
        if seqNoise is not None:
            seqNoise = filter_distributed(seqNoise)
        print(
            f'Current worker files: {len(seqTrain)} train, {len(seqVal)} val')


    print("")
    print(f'Loading audio data at {args.pathDB}')
    print("Loading the training dataset")
    trainDataset = AudioBatchData(args.pathDB,
                                  args.sizeWindow,
                                  seqTrain,
                                  phoneLabels,
                                  len(speakers),
                                  nProcessLoader=args.n_process_loader,
                                  MAX_SIZE_LOADED=args.max_size_loaded,
                                  augment_future=args.augment_future,
                                  augment_past=args.augment_past,
                                  augmentation=augmentation_factory(args, noiseDataset))
    print("Training dataset loaded")
    print("")

    if seqVal:
        print("Loading the validation dataset")
        valDataset = AudioBatchData(args.pathDB,
                                args.sizeWindow,
                                seqVal,
                                phoneLabels,
                                len(speakers),
                                nProcessLoader=args.n_process_loader)
        print("Validation dataset loaded")
        print("")
    else:
        valDataset = None

    if args.load is not None:
        cpcModel, args.hiddenGar, args.hiddenEncoder = \
            fl.loadModel(args.load)

    else:
        # Encoder network
        encoderNet = fl.getEncoder(args)
        # AR Network
        arNet = fl.getAR(args)

        if args.cpc_mode == "bert":
            cpcModel = model.CPCBertModel(encoderNet, arNet,
                                          blockSize=args.nPredicts)
            cpcModel.supervised = args.supervised
        else:
            cpcModel = model.CPCModel(encoderNet, arNet)

    cpcModel.supervised = args.supervised

    # Training criterion
    if args.load is not None and args.loadCriterion:
        cpcCriterion = loadCriterion(args.load[0], cpcModel.gEncoder.DOWNSAMPLING,
                                     len(speakers), nPhones)
    else:
        cpcCriterion = getCriterion(args, cpcModel.gEncoder.DOWNSAMPLING,
                                    len(speakers), nPhones)

    if load_optimizer:
        state_dict = torch.load(args.load[0], 'cpu')
        cpcCriterion.load_state_dict(state_dict["cpcCriterion"])

    cpcCriterion.cuda()
    cpcModel.cuda()

    # Optimizer
    g_params = list(cpcCriterion.parameters()) + list(cpcModel.parameters())

    clustering = None
    if args.clustering is not None:
        if args.clustering == 'deepClustering':
            clustering = DeepClustering(args.n_clusters, args.hiddenGar,
                                        args.cluster_delay,
                                        args.cluster_iter,
                                        args.clustering_update).cuda()
            g_params += list(clustering.parameters())
        elif args.clustering == 'deepEmbedded':
            clustering = DeepEmbeddedClustering(args.learningRate,
                                                args.n_clusters,
                                                args.hiddenGar,
                                                args.cluster_delay,
                                                args.cluster_iter,
                                                args.clustering_update).cuda()
        elif args.clustering == 'CTCClustering':
            clustering = CTCCLustering(args.n_clusters, args.hiddenGar,
                                       args.cluster_delay,
                                       args.cluster_iter,
                                       args.clustering_update).cuda()
        clustering = torch.nn.DataParallel(clustering,
                                           device_ids=range(args.nGPU))

    lr = args.learningRate
    optimizer = torch.optim.Adam(g_params, lr=lr,
                                 betas=(args.beta1, args.beta2),
                                 eps=args.epsilon)

    if load_optimizer:
        print("Loading optimizer " + args.load[0])
        state_dict = torch.load(args.load[0], 'cpu')
        if "optimizer" in state_dict:
            optimizer.load_state_dict(state_dict["optimizer"])

    # Checkpoint
    if args.pathCheckpoint is not None:
        if not os.path.isdir(args.pathCheckpoint):
            os.mkdir(args.pathCheckpoint)
        args.pathCheckpoint = os.path.join(args.pathCheckpoint, "checkpoint")
        if args.is_local_master:
            with open(args.pathCheckpoint + "_args.json", 'w') as file:
                json.dump(vars(args), file, indent=2)

    scheduler = None
    if args.schedulerStep > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    args.schedulerStep,
                                                    gamma=0.5)
    if args.schedulerRamp is not None:
        n_epoch = args.schedulerRamp
        print(f"Ramp activated. n_e = {n_epoch}")
        scheduler_ramp = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                           lr_lambda=lambda epoch: utils.ramp_scheduling_function(
                                                               n_epoch, epoch),
                                                           last_epoch=-1)
        if scheduler is None:
            scheduler = scheduler_ramp
        else:
            scheduler = utils.SchedulerCombiner([scheduler_ramp, scheduler],
                                                [0, args.schedulerRamp])
    if scheduler is not None:
        for i in range(len(logs["epoch"])):
            scheduler.step()

    print('args.local_rank: ' + str(args.local_rank))
    if args.distributed:
        cpcModel = torch.nn.parallel.DistributedDataParallel(cpcModel, device_ids=[
                                                             args.local_rank], output_device=args.local_rank, broadcast_buffers=True)
        cpcCriterion = torch.nn.parallel.DistributedDataParallel(cpcCriterion, device_ids=[
                                                                 args.local_rank], output_device=args.local_rank, broadcast_buffers=True)
    else:
        cpcModel = torch.nn.DataParallel(cpcModel,
                                         device_ids=range(args.nGPU)).cuda()
        cpcCriterion = torch.nn.DataParallel(cpcCriterion,
                                             device_ids=range(args.nGPU)).cuda()

    adversarial = None
    if args.adversarial:
        adversarial = cr.AdvSpeakerCriterion(args.hiddenGar,
                                             len(speakers), args.onEncoder)
        adversarial = torch.nn.DataParallel(adversarial,
                                            device_ids=range(args.nGPU))
        adversarial.cuda()

    balance_sampler = None
    if args.balance_type is not None:
        balance_sampler= get_balance_sampler(args.balance_type,
                                             balance_coeff=args.balance_coeff)

    run(trainDataset,
        valDataset,
        batchSize,
        args.samplingType,
        balance_sampler,
        cpcModel,
        cpcCriterion,
        args.nEpoch,
        args.pathCheckpoint if args.is_local_master else None,
        optimizer,
        scheduler,
        logs,
        adversarial,
        clustering)


def parseArgs(argv, defaults=None):
    # Run parameters
    parser = argparse.ArgumentParser(description='Trainer')

    # Default arguments:
    parser = set_default_cpc_config(parser)

    group_db = parser.add_argument_group('Dataset')
    group_db.add_argument('--pathDB', type=str, default=None,
                          help='Path to the directory containing the '
                          'data.')
    group_db.add_argument('--file_extension', type=str, default=".flac",
                          help="Extension of the audio files in the dataset.")
    group_db.add_argument('--pathTrain', type=str, default=None,
                          help='Path to a .txt file containing the list of the '
                          'training sequences.')
    group_db.add_argument('--pathVal', type=str, default=None,
                          help='Path to a .txt file containing the list of the '
                          'validation sequences.')
    group_db.add_argument('--n_process_loader', type=int, default=8,
                          help='Number of processes to call to load the '
                          'dataset')
    group_db.add_argument('--ignore_cache', action='store_true',
                          help='Activate if the dataset has been modified '
                          'since the last training session.')
    group_db.add_argument('--path_cache', type=str,default=None,
                          help="For big datasets, path to an existing cache")
    group_db.add_argument('--max_size_loaded', type=int, default=4000000000,
                          help='Maximal amount of data (in byte) a dataset '
                          'can hold in memory at any given time')
    group_db.add_argument('--balance_type', type=str, default=None,
                         choices=['linear', 'log', 'pow'])
    group_db.add_argument('--balance_coeff', type=float, default=0.5)
    group_supervised = parser.add_argument_group(
        'Supervised mode (depreciated)')
    group_supervised.add_argument('--supervised', action='store_true',
                                  help='(Depreciated) Disable the CPC loss and activate '
                                  'the supervised mode. By default, the supervised '
                                  'training method is the speaker classification.')
    group_supervised.add_argument('--pathPhone', type=str, default=None,
                                  help='(Supervised mode only) Path to a .txt '
                                  'containing the phone labels of the dataset. If given '
                                  'and --supervised, will train the model using a '
                                  'phone classification task.')
    group_supervised.add_argument('--CTC', action='store_true')

    group_save = parser.add_argument_group('Save')
    group_save.add_argument('--pathCheckpoint', type=str, default=None,
                            help="Path of the output directory.")
    group_save.add_argument('--logging_step', type=int, default=1000)
    group_save.add_argument('--save_step', type=int, default=5,
                            help="Frequency (in epochs) at which a checkpoint "
                            "should be saved")

    group_load = parser.add_argument_group('Load')
    group_load.add_argument('--load', type=str, default=None, nargs='*',
                            help="Load an exsiting checkpoint. Should give a path "
                            "to a .pt file. The directory containing the file to "
                            "load should also have a 'checkpoint.logs' and a "
                            "'checkpoint.args'")
    group_load.add_argument('--loadCriterion', action='store_true',
                            help="If --load is activated, load the state of the "
                            "training criterion as well as the state of the "
                            "feature network (encoder + AR)")
    group_load.add_argument('--restart', action='store_true',
                            help="If any checkpoint is found, ignore it and "
                            "restart the training from scratch.")

    group_gpu = parser.add_argument_group('GPUs')
    group_gpu.add_argument('--nGPU', type=int, default=-1,
                           help="Number of GPU to use (default: use all "
                           "available GPUs)")
    group_gpu.add_argument('--batchSizeGPU', type=int, default=8,
                           help='Number of batches per GPU.')
    parser.add_argument('--debug', action='store_true',
                        help="Load only a very small amount of files for "
                        "debugging purposes.")

    group_distrubed = parser.add_argument_group(
        'Distributed training (FAIR only)')
    group_distrubed.add_argument('--distributed', action='store_true')
    group_distrubed.add_argument("--local_rank", type=int, default=-1,
                                 help="Multi-GPU - Local rank")
    group_distrubed.add_argument("--master_port", type=int, default=-1,
                                 help="Master port (for multi-node SLURM jobs)")
    if defaults is not None:
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)

    if args.pathDB is None and (args.pathCheckpoint is None or args.restart):
        parser.print_help()
        print("Either provides an input dataset or a checkpoint to load")
        sys.exit()

    assert args.bandreject_scaler >= 0

    if args.pathCheckpoint is not None:
        args.pathCheckpoint = os.path.abspath(args.pathCheckpoint)

    if args.load is not None:
        args.load = [os.path.abspath(x) for x in args.load]

    # set it up if needed, so that it is dumped along with other args
    if args.random_seed is None:
        args.random_seed = random.randint(0, 2**31)

    if args.nGPU < 0:
        args.nGPU = torch.cuda.device_count()
    assert args.nGPU <= torch.cuda.device_count(),\
        f"number of GPU asked: {args.nGPU}," \
        f"number GPU detected: {torch.cuda.device_count()}"
    print(f"Let's use {args.nGPU} GPUs!")

    if args.arMode == 'no_ar':
        args.hiddenGar = args.hiddenEncoder
    return args

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    args = sys.argv[1:]
    main(args)
