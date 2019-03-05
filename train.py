import os
from random import shuffle
import torch

from dataset import AudioBatchData
from model import CPCModel
from criterion import CPCUnsupersivedCriterion, SpeakerCriterion, PhoneCriterion

import json
import numpy as np
import argparse

import visdom
vis = visdom.Visdom()


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
    output = {"step": 160}
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
    model.train()
    cpcCriterion.train()
    if scheduler is not None:
        scheduler.step()

    nGtSequenceByGPU = cpcCriterion.nGtSequence // len(model.device_ids)
    logs = {"step": 0}

    for step, fulldata in enumerate(dataLoader):

        optimizer.zero_grad()

        batchData, label = fulldata
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
            cpcCriterion):
    model.eval()
    cpcCriterion.eval()

    logs = {"step": 0}
    nGtSequenceByGPU = cpcCriterion.nGtSequence // len(model.device_ids)
    for step, fulldata in enumerate(dataLoader):

        batchData, label = fulldata
        batchData = batchData.cuda()
        label = label.cuda()
        cFeature, gtPredictions, otherEncoded = model(batchData,
                                                      nAR=nGtSequenceByGPU)

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
    windowToken = None

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
        windowToken = publishLogs(
            logs, name="CPC validation", window_tokens=windowToken)

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
                        choices=['speaker', 'uniform', 'sequence'])
    parser.add_argument('--nGPU', type=int, default=1)
    parser.add_argument('--batchSizeGPU', type=int, default=8)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

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

    cpcModel = CPCModel(args.hiddenEncoder, args.hiddenGar)

    if args.load != "":
        print("Loading checkpoint " + args.load)
        state_dict = torch.load(args.load)
        cpcModel.load_state_dict(state_dict["gEncoder"])

    nGPU = torch.cuda.device_count() if args.nGPU == -1 else args.nGPU
    assert nGPU <= torch.cuda.device_count(), f"number of GPU asked: {nGPU}," \
        f"number GPU detected: {torch.cuda.device_count()}"

    batchSize = nGPU * args.batchSizeGPU
    nGtSequence = nGPU
    print("Let's use", nGPU, "GPUs!")
    cpcModel = torch.nn.DataParallel(cpcModel, device_ids=range(nGPU))

    if not args.supervised:
        cpcCriterion = CPCUnsupersivedCriterion(args.nPredicts, args.hiddenGar,
                                                args.hiddenEncoder,
                                                args.negativeSamplingExt,
                                                nGtSequence)
    elif args.pathPhone is not None:
        cpcCriterion = PhoneCriterion(args.hiddenGar, nPhones)
    else:
        cpcCriterion = SpeakerCriterion(
            args.hiddenGar, trainDataset.getNSpeakers(), 1)

    optimizeModel = not args.eval

    if args.eval:
        print("Evaluation mode")

    cpcCriterion.cuda()
    cpcModel.cuda()

    # Optimizer
    g_params = list(cpcCriterion.parameters())

    if optimizeModel:
        print("Optimizing model")
        g_params += list(cpcModel.parameters())

    optimizer = torch.optim.Adam(g_params, lr=args.learningRate)

    if args.schedulerStep > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.schedulerStep)
    else:
        scheduler = None

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
                                                  args.pathPhone is None),
                                              num_workers=2)
    valLoader = torch.utils.data.DataLoader(valDataset,
                                            batch_sampler=valDataset.getSampler(
                                                batchSize, args.groupSize,
                                                args.samplingType, False),
                                            num_workers=2)

    run(trainLoader,
        valLoader,
        cpcModel,
        cpcCriterion,
        args.nEpoch,
        args.pathCheckpoint,
        optimizer,
        scheduler)
