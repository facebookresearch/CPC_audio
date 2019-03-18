import argparse
import json
import os
from random import shuffle

import numpy as np
import torch

from dataset import AudioBatchData
from model import CPCModel
from criterion import CPCUnsupersivedCriterion, SpeakerCriterion, \
                      PhoneCriterion


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


def saveLogs(data, pathLogs):
    with open(pathLogs, 'w') as file:
        json.dump(data, file, indent=2)


def findAllSeqs(dbPath):
    speakers = [f for f in os.listdir(dbPath)
                if os.path.isdir(os.path.join(dbPath, f))]
    outSeqs = []
    for speaker in speakers:
        refPath = os.path.join(dbPath, speaker)
        chapters = os.listdir(refPath)
        for chapter in chapters:
            fullPath = os.path.join(refPath, chapter)
            outSeqs += [f for f in os.listdir(fullPath)
                        if os.path.splitext(f)[1] == '.flac']

    return outSeqs


def parseTxtSplit(pathTxt):
    return [p.replace('\n', '') + ".flac" for p in
            open(pathTxt, 'r').readlines()]


def parseSeqLabels(pathLabels):
    lines = open(pathLabels, 'r').readlines()
    output = {"step": 160}  # Step in librispeech dataset is 160bits
    maxPhone = 0
    for line in lines:
        data = line.split()
        output[data[0]] = [int(x) for x in data[1:]]
        maxPhone = max(maxPhone, max(output[data[0]]))
    return output, maxPhone + 1


def trainStep(dataLoader,
              model,
              cpcCriterion,
              optimizer,
              scheduler):
    if model.optimize:
        model.train()
    cpcCriterion.train()
    if scheduler is not None:
        scheduler.step()

    logs = {"step": 0}
    for step, fulldata in enumerate(dataLoader):

        batchData, label = fulldata
        batchData = batchData.cuda()
        label = label.cuda()
        cFeature, encodedData = model(batchData)

        allLosses, allAcc = cpcCriterion(cFeature, encodedData, label)

        totLoss = allLosses.sum()
        totLoss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if "locLoss_train" not in logs:
            logs["locLoss_train"] = np.zeros(allLosses.size(1))
            logs["locAcc_train"] = np.zeros(allLosses.size(1))

        logs["step"] += 1
        logs["locLoss_train"] += (allLosses.mean(dim=0)).detach().cpu().numpy()
        logs["locAcc_train"] += (allAcc.mean(dim=0)).cpu().numpy()

    updateAndShowLogs("Update %d, training loss:" %
                      (logs["step"] + 1), logs, logs["locLoss_train"].shape[0])
    return logs


def valStep(dataLoader,
            model,
            cpcCriterion):
    model.eval()
    cpcCriterion.eval()

    logs = {"step": 0}
    for step, fulldata in enumerate(dataLoader):

        batchData, label = fulldata

        batchData = batchData.cuda()
        label = label.cuda()
        cFeature, encodedData = model(batchData)

        allLosses, allAcc = cpcCriterion(cFeature, encodedData, label)

        if "locLoss_val" not in logs:
            logs["locLoss_val"] = np.zeros(allLosses.size(1))
            logs["locAcc_val"] = np.zeros(allLosses.size(1))

        logs["step"] += 1
        logs["locLoss_val"] += allLosses.mean(dim=0).detach().cpu().numpy()
        logs["locAcc_val"] += allAcc.mean(dim=0).cpu().numpy()

    logs["step"] = step
    updateAndShowLogs("Validation loss:", logs, logs["locLoss_val"].shape[0])
    return logs


def run(trainLoader,
        valLoader,
        cpcModel,
        cpcCriterion,
        nEpoch,
        pathCheckpoint,
        optimizer,
        scheduler):

    print("Running %d epochs" % nEpoch)

    #  Logs
    logs = {"epoch": []}

    for epoch in range(nEpoch):

        print("Starting epoch %d" % epoch)
        print("Training dataset %d batches, Validation dataset %d batches" %
              (len(trainLoader), len(valLoader)))

        locLogsTrain = trainStep(
            trainLoader, cpcModel, cpcCriterion, optimizer, scheduler)

        locLogsVal = valStep(valLoader, cpcModel, cpcCriterion)

        for key, value in dict(locLogsTrain, **locLogsVal).items():
            if key not in logs:
                logs[key] = [None for x in range(epoch)]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            logs[key].append(value)

        logs["epoch"].append(epoch)

        # Dirty checkpoint save
        if pathCheckpoint is not None:
            stateDict = {"gEncoder": cpcModel.module.state_dict(),
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
    parser.add_argument('--pathVal', type=str,
                        default="/datasets01/LibriSpeech/022219/LibriSpeech100_labels_split/test_split.txt")
    parser.add_argument('--pathPhone', type=str, default=None)
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
    parser.add_argument('--schedulerStep', type=int,
                        default=0)
    parser.add_argument('--groupSize', type=int, default=2)
    parser.add_argument('--samplingType', type=str, default='speaker',
                        choices=['speaker', 'uniform',
                                 'sequence', 'sequential'])
    parser.add_argument('--nGPU', type=int, default=-1)
    parser.add_argument('--batchSizeGPU', type=int, default=8)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    # Datasets
    if args.pathTrain is None:
        seqNames = findAllSeqs(args.pathDB)
    else:
        seqNames = parseTxtSplit(args.pathTrain)

    if args.pathVal is None:
        shuffle(seqNames)
        sizeTrain = int(0.8 * len(seqNames))
        seqTrain, seqVal = seqNames[:sizeTrain], seqNames[sizeTrain:]
    else:
        seqTrain = seqNames
        seqVal = parseTxtSplit(args.pathVal)

    if args.debug:
        seqTrain = seqTrain[:2000]
        seqVal = seqVal[:2000]

    phoneLabels = None
    if args.supervised and args.pathPhone is not None:
        print("Loading the phone labels at " + args.pathPhone)
        phoneLabels, nPhones = parseSeqLabels(args.pathPhone)

    trainDataset = AudioBatchData(args.pathDB,
                                  args.sizeWindow,
                                  seqTrain,
                                  phoneLabels)

    valDataset = AudioBatchData(args.pathDB,
                                args.sizeWindow,
                                seqVal,
                                phoneLabels)

    # Base Model
    cpcModel = CPCModel(args.hiddenEncoder, args.hiddenGar,
                        args.samplingType == "sequential")

    if args.load != "":
        print("Loading checkpoint " + args.load)
        state_dict = torch.load(args.load)
        cpcModel.load_state_dict(state_dict["gEncoder"])

    nGPU = torch.cuda.device_count() if args.nGPU == -1 else args.nGPU
    assert nGPU <= torch.cuda.device_count(), f"number of GPU asked: {nGPU}," \
        f"number GPU detected: {torch.cuda.device_count()}"

    batchSize = nGPU * args.batchSizeGPU
    print("Let's use", nGPU, "GPUs!")
    cpcModel = torch.nn.DataParallel(cpcModel, device_ids=range(nGPU))

    # Training criterion
    if not args.supervised:
        cpcCriterion = CPCUnsupersivedCriterion(args.nPredicts, args.hiddenGar,
                                                args.hiddenEncoder,
                                                args.negativeSamplingExt)
    elif args.pathPhone is not None:
        cpcCriterion = PhoneCriterion(args.hiddenGar, nPhones)
    else:
        cpcCriterion = SpeakerCriterion(args.hiddenGar,
                                        trainDataset.getNSpeakers())

    cpcCriterion = torch.nn.DataParallel(cpcCriterion, device_ids=range(nGPU))
    cpcModel.optimize = True
    if args.eval:
        print("Evaluation mode")
        cpcModel.optimize = False
        cpcModel.eval()
        for g in cpcModel.parameters():
            g.requires_grad = False

    cpcCriterion.cuda()
    cpcModel.cuda()

    # Optimizer
    g_params = list(cpcCriterion.parameters())

    if not args.eval:
        print("Optimizing model")
        g_params += list(cpcModel.parameters())

    optimizer = torch.optim.Adam(g_params, lr=args.learningRate)

    # Scheduler
    if args.schedulerStep > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.schedulerStep, gamma=0.3)
    else:
        scheduler = None

    # Checkpoint
    if args.pathCheckpoint is not None:
        if not os.path.isdir(args.pathCheckpoint):
            os.mkdir(args.pathCheckpoint)
        args.pathCheckpoint = os.path.join(args.pathCheckpoint, "checkpoint")
        with open(args.pathCheckpoint + "_args.json", 'w') as file:
            json.dump(vars(args), file, indent=2)

    trainLoader = torch.utils.data.DataLoader(trainDataset,
                                              batch_sampler=trainDataset.getSampler(
                                                  batchSize, args.groupSize,
                                                  args.samplingType,
                                                  True),
                                              num_workers=nGPU)
    valLoader = torch.utils.data.DataLoader(valDataset,
                                            batch_sampler=valDataset.getSampler(
                                                batchSize, args.groupSize,
                                                args.samplingType, False),
                                            num_workers=nGPU)
    run(trainLoader,
        valLoader,
        cpcModel,
        cpcCriterion,
        args.nEpoch,
        args.pathCheckpoint,
        optimizer,
        scheduler)
