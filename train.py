import torch
from torch import nn
from torch.utils.data import Subset

from dataset import AudioBatchData, AudioBatchDataset
from model import CPCModel
from criterion import CPCUnsupersivedCriterion, SpeakerCriterion

import json
import numpy as np
import argparse

import visdom
vis = visdom.Visdom()

N_GT_SEQUENCE_BY_GPU = 1
BATCH_SIZE_BY_GPU = 8


def updateAndShowLogs(text, logs, nPredicts):
    logStep = logs["step"]

    print("")
    print('-'*50)
    print(text)
    strSteps = ['Step'] + [str(s) for s in range(1, nPredicts + 1)]
    formatCommand = ' '.join(['{:>16}' for x in range(nPredicts + 1)])
    print(formatCommand.format(*strSteps))

    for key in logs:

        if key == "step":
            continue

        logs[key] /= logStep
        strLog = [key] + ["{:10.6f}".format(s) for s in logs[key]]
        print(formatCommand.format(*strLog))

    print('-'*50)


def publishLogs(data, name="", window_tokens=None, env="main"):
    if window_tokens is None:
        window_tokens = {key: None for key in data}

    for key, plot in data.items():

        if key in ("step", "epoch"):
            continue

        nItems = len(plot)
        inputY = np.array([plot[x] for x in range(nItems) if 0 is not None])
        inputX = np.array([data["epoch"][x]
                           for x in range(nItems) if plot[x] is not None])

        opts = {'title': name + " " + key,
                'legend': [str(x) for x in range(len(plot[0]))],
                'xlabel': 'epoch', 'ylabel': 'loss'}

        window_tokens[key] = vis.line(X=inputX, Y=inputY, opts=opts,
                                      win=window_tokens[key], env=env)

    return window_tokens


def saveLogs(data, pathLogs):

    with open(pathLogs, 'w') as file:
        json.dump(data, file, indent=2)


def makeOptimizer(optimizer_name, g_params, lr, **kwargs):
    if optimizer_name == 'adam':
        from torch.optim import Adam
        return Adam(g_params, lr=lr, **kwargs)
    else:
        raise ValueError('{} is not a supported optimizer'
                         .format(optimizer_name))


def makeScheduler(scheduler_name, optimizer, **kwargs):
    if scheduler_name == 'step_lr':
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, step_size=1, **kwargs)
    else:
        raise ValueError('{} is not a supported scheduler'
                         .format(scheduler_name))


def trainStep(dataLoader,
              model,
              cpcCriterion,
              optimizer,
              scheduler,
              nGtSequenceByGPU):
    model.train()
    cpcCriterion.train()
    if scheduler:
        scheduler.step()

    logs = {"step": 0}

    for step, fulldata in enumerate(dataLoader):

        optimizer.zero_grad()

        batchData, label = fulldata

        if batchData.size(0) <= cpcCriterion.nGtSequence:
            continue

        batchData = batchData.cuda()
        label = label.cuda()
        cFeature, gtPredictions, otherEncoded = model(
            batchData, nAR=nGtSequenceByGPU)

        allLosses, allAcc = cpcCriterion(
            cFeature, gtPredictions, otherEncoded, label)

        totLoss = allLosses.sum()
        totLoss.backward()
        optimizer.step()

        if "locLoss_train" not in logs:
            logs["locLoss_train"] = np.zeros(allLosses.size(0))
            logs["locAcc_train"] = np.zeros(allLosses.size(0))

        logs["step"] += 1
        logs["locLoss_train"] += allLosses.detach().cpu().numpy()
        logs["locAcc_train"] += allAcc.cpu().numpy()

    updateAndShowLogs("Update %d, training loss:" %
                      (logs["step"] + 1), logs, logs["locLoss_train"].shape[0])
    return logs


def valStep(dataLoader,
            model,
            cpcCriterion,
            nGtSequenceByGPU):
    model.eval()
    cpcCriterion.eval()

    logs = {"step": 0}
    for step, fulldata in enumerate(dataLoader):

        batchData, label = fulldata

        if batchData.size(0) <= cpcCriterion.nGtSequence:
            continue

        batchData = batchData.cuda()
        label = label.cuda()
        cFeature, gtPredictions, otherEncoded = model(batchData,
                                                      nAR=nGtSequenceByGPU)

        if otherEncoded.size() == 0:
            print(batchData.size())

        allLosses, allAcc = cpcCriterion(
            cFeature, gtPredictions, otherEncoded, label)

        if "locLoss_val" not in logs:
            logs["locLoss_val"] = np.zeros(allLosses.size(0))
            logs["locAcc_val"] = np.zeros(allLosses.size(0))

        logs["step"] += 1
        logs["locLoss_val"] += allLosses.detach().cpu().numpy()
        logs["locAcc_val"] += allAcc.cpu().numpy()

    logs["step"] = step
    updateAndShowLogs("Validation loss:", logs, logs["locLoss_val"].shape[0])

    return logs


def run(trainDataset,
        valDataset,
        cpcModel,
        cpcCriterion,
        nEpoch,
        batchSize,
        pathCheckpoint,
        optimizer,
        scheduler,
        nGtSequenceByGPU):
    print("Dataset size: %d bits, running %d epochs" %
          (len(audioData), nEpoch))

    #  Logs
    logs = {"epoch": []}
    windowToken = None

    for epoch in range(nEpoch):

        print("Starting epoch %d" % epoch)

        print("Training dataset %d samples, Validation dataset %d samples" %
              (len(trainDataset), len(valDataset)))

        trainLoader = torch.utils.data.DataLoader(trainDataset,
                                                  batch_size=batchSize,
                                                  shuffle=True,
                                                  num_workers=2)

        locLogsTrain = trainStep(
            trainLoader, cpcModel, cpcCriterion, optimizer, scheduler,
            nGtSequenceByGPU)

        valLoader = torch.utils.data.DataLoader(valDataset,
                                                batch_size=batchSize,
                                                shuffle=False,
                                                num_workers=2)

        locLogsVal = valStep(valLoader, cpcModel,
                             cpcCriterion, nGtSequenceByGPU)

        for key, value in dict(locLogsTrain, **locLogsVal).items():
            if key not in logs:
                logs[key] = [None for x in range(epoch)]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            logs[key].append(value)

        logs["epoch"].append(epoch)
        windowToken = publishLogs(
            logs, name="CPC validation", window_tokens=windowToken)

        # Dirty checkpoint save
        if pathCheckpoint is not None:
            stateDict = {"gEncoder": cpcModel.state_dict(),
                         "cpcCriterion": cpcCriterion.state_dict()}

            torch.save(stateDict, pathCheckpoint + "_" + str(epoch) + '.pt')
            saveLogs(logs, pathCheckpoint + "_logs.json")


if __name__ == "__main__":

    # Run parameters
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument(
        '--pathDB', type=str, default="/datasets01/LibriSpeech/022219/train-clean-100/")
    parser.add_argument('--pathTrain', type=str,
                        default="/datasets01/LibriSpeech/022219/LibriSpeech100_labels_split/train_split.txt")
    parser.add_argument('--pathVal', type=str, default=None)
    parser.add_argument('--hiddenEncoder', type=int, default=512)
    parser.add_argument('--hiddenGar', type=int, default=256)
    parser.add_argument('--nPredicts', type=int, default=12)
    parser.add_argument('--negativeSamplingExt', type=int, default=128)
    parser.add_argument('--supervised', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--load', type=str, default="")
    parser.add_argument('--learningRate', type=float, default=2e-4)
    parser.add_argument('--pathCheckpoint', type=str, default=None)
    parser.add_argument('--sizeWindow', type=int, default=20480)
    parser.add_argument('--nEpoch', type=int, default=10)
    parser.add_argument('--optimizerName', type=str,
                        default='adam', choices=['adam'])
    parser.add_argument('--schedulerName', type=str,
                        default=None, choices=[None, 'step_lr'])
    parser.add_argument('--nGPU', type=int, default=1)

    args = parser.parse_args()

    audioData = AudioBatchData(args.pathDB, seqNamesPath=args.pathTrain)

    if args.pathVal is None:
        baseDataset = AudioBatchDataset(audioData, args.sizeWindow)
        sizeDataset = len(baseDataset)
        sizeTrain = int(0.8 * sizeDataset)

        indices = torch.randperm(sizeDataset)
        trainDataset = Subset(baseDataset, indices[:sizeTrain])
        valDataset = Subset(baseDataset, indices[sizeTrain:])
    else:
        trainDataset = AudioBatchDataset(audioData, args.sizeWindow)
        valData = AudioBatchData(args.pathDB, seqNamesPath=args.pathVal)
        valDataset = AudioBatchDataset(valData, args.sizeWindow)

    nGPU = torch.cuda.device_count() if args.nGPU == -1 else args.nGPU
    assert nGPU <= torch.cuda.device_count(), f"number of GPU asked: {nGPU}," \
        f"number GPU detected: {torch.cuda.device_count()}"

    batchSize = nGPU * BATCH_SIZE_BY_GPU
    nGtSequenceByGPU = -1 if args.supervised else N_GT_SEQUENCE_BY_GPU
    nGtSequence = nGPU * nGtSequenceByGPU

    cpcModel = CPCModel(args.hiddenEncoder, args.hiddenGar)
    if nGPU > 1:
        print("Let's use", nGPU, "GPUs!")
        cpcModel = nn.DataParallel(cpcModel, device_ids=range(nGPU))
    else:
        print("Let's use", nGPU, "GPU!")

    if args.load != "":
        print("Loading checkpoint " + args.load)
        state_dict = torch.load(args.load)
        cpcModel.load_state_dict(state_dict["gEncoder"])

    if args.supervised:
        cpcCriterion = SpeakerCriterion(
            args.hiddenGar, audioData.getNSpeakers(), 1)

    else:
        cpcCriterion = CPCUnsupersivedCriterion(args.nPredicts, args.hiddenGar,
                                                args.hiddenEncoder,
                                                args.negativeSamplingExt,
                                                nGtSequence)

    optimizeModel = not args.eval

    if args.eval:
        print("Evaluation mode")
        args.pathCheckpoint = None

    cpcCriterion.cuda()
    cpcModel.cuda()

    # Optimizer
    g_params = list(cpcCriterion.parameters())

    if optimizeModel:
        print("Optimizing model")
        g_params += list(cpcModel.parameters())

    # Nombre magique
    optimizer = makeOptimizer(
        args.optimizerName, g_params, lr=args.learningRate)

    if args.schedulerName:
        scheduler = makeScheduler(args.schedulerName, optimizer)
    else:
        scheduler = None

    run(trainDataset,
        valDataset,
        cpcModel,
        cpcCriterion,
        args.nEpoch,
        batchSize,
        args.pathCheckpoint,
        optimizer,
        scheduler,
        nGtSequenceByGPU)
