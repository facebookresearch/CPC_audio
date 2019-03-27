import argparse
import json
import os
from random import shuffle
import sys

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


def loadArgs(args, locArgs):
    for k, v in locArgs.items():
        setattr(args, k, v)


def getCheckpointData(pathDir):
    checkpoints = [x for x in os.listdir(pathDir)
                   if os.path.splitext(x)[1] == '.pt']
    if len(checkpoints) == 0:
        print("No checkpoints found at " + pathDir)
        return None
    checkpoints.sort(key = lambda x : int(os.path.splitext(x[11:])[0]))
    data = os.path.join(pathDir, checkpoints[-1])
    with open(os.path.join(pathDir, 'checkpoint_logs.json'), 'rb') as file:
        logs = json.load(file)

    with open(os.path.join(pathDir, 'checkpoint_args.json'), 'rb') as file:
        args = json.load(file)

    return data, logs, args


def findAllSeqs(dirName,
                recursionLevel=2,
                extension='.flac'):

    dirName = os.path.join(dirName, '')
    dirList = [dirName]
    prefixSize = len(dirName)

    for recursion in range(recursionLevel):
        nextList = []
        for item in dirList:
            nextList += [os.path.join(item, f) for f in os.listdir(item)
                         if os.path.isdir(os.path.join(item, f))]
        dirList = nextList

    outSequences = []
    for directory in dirList:
        basePath = directory[prefixSize:]
        speaker = int(os.path.normpath(basePath).split(os.sep)[0])
        for item in os.listdir(directory):
            if os.path.splitext(item)[1] != extension:
                continue
            outSequences.append((speaker, os.path.join(basePath, item)))

    return outSequences


def filterSeqs(pathTxt, allSeqs):
    inSeqs = [p.replace('\n', '') for p in open(pathTxt, 'r').readlines()]
    return [x for x in allSeqs
            if os.path.splitext(os.path.basename(x[1]))[0] in inSeqs]


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
              optimizer):

    if model.optimize:
        model.train()
    cpcCriterion.train()

    logs = {"step": 0}
    for step, fulldata in enumerate(dataLoader):

        batchData, label = fulldata
        batchData = batchData.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
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

        batchData = batchData.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
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
        logs):

    print("Running %d epochs" % nEpoch)
    startEpoch = len(logs["epoch"])
    for epoch in range(startEpoch, nEpoch):

        print("Starting epoch %d" % epoch)
        print("Training dataset %d batches, Validation dataset %d batches" %
              (len(trainLoader), len(valLoader)))

        locLogsTrain = trainStep(
            trainLoader, cpcModel, cpcCriterion, optimizer)

        locLogsVal = valStep(valLoader, cpcModel, cpcCriterion)

        for key, value in dict(locLogsTrain, **locLogsVal).items():
            if key not in logs:
                logs[key] = [None for x in range(epoch)]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            logs[key].append(value)

        logs["epoch"].append(epoch)

        if pathCheckpoint is not None \
                and (epoch % 5 == 0 or epoch == nEpoch-1):
            stateDict = {"gEncoder": cpcModel.module.state_dict(),
                         "cpcCriterion": cpcCriterion.state_dict()}

            torch.save(stateDict, pathCheckpoint + "_" + str(epoch) + '.pt')
            saveLogs(logs, pathCheckpoint + "_logs.json")


def main(args):
    print(f'CONFIG:\n{json.dumps(vars(args), indent=4, sort_keys=True)}')
    print('-' * 50)

    logs = {"epoch": []}
    if args.pathCheckpoint is not None and os.path.isdir(args.pathCheckpoint)\
            and not args.restart:
        checkpointData = getCheckpointData(args.pathCheckpoint)
        if checkpointData is not None:
            load, logs, locArgs = checkpointData
            loadArgs(args, locArgs)
            args.load = load

    seqNames = findAllSeqs(args.pathDB,
                           recursionLevel=args.dataset_levels,
                           extension=args.file_extension)

    # Datasets
    if args.pathTrain is not None:
        seqTrain = filterSeqs(args.pathTrain, seqNames)
    else:
        seqTrain = seqNames

    if args.pathVal is None:
        shuffle(seqTrain)
        sizeTrain = int(0.8 * len(seqTrain))
        seqTrain, seqVal = seqTrain[:sizeTrain], seqTrain[sizeTrain:]
    else:
        seqVal = filterSeqs(args.pathVal, seqNames)

    if args.debug:
        seqTrain = seqTrain[:2000]
        seqVal = seqVal[:2000]

    phoneLabels = None
    if args.supervised and args.pathPhone is not None:
        print("Loading the phone labels at " + args.pathPhone)
        phoneLabels, nPhones = parseSeqLabels(args.pathPhone)

    print(f'Loading audio data at {args.pathDB}')
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

    if args.nGPU < 0:
        args.nGPU = torch.cuda.device_count()
    assert args.nGPU <= torch.cuda.device_count(), f"number of GPU asked: {args.nGPU}," \
        f"number GPU detected: {torch.cuda.device_count()}"

    batchSize = args.nGPU * args.batchSizeGPU
    print("Let's use", args.nGPU, "GPUs!")
    cpcModel = torch.nn.DataParallel(cpcModel, device_ids=range(args.nGPU))

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

    cpcCriterion = torch.nn.DataParallel(cpcCriterion,
                                         device_ids=range(args.nGPU))
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

    # Checkpoint
    if args.pathCheckpoint is not None:
        if not os.path.isdir(args.pathCheckpoint):
            os.mkdir(args.pathCheckpoint)
        args.pathCheckpoint = os.path.join(args.pathCheckpoint, "checkpoint")
        with open(args.pathCheckpoint + "_args.json", 'w') as file:
            json.dump(vars(args), file, indent=2)

    trainLoader = torch.utils.data.DataLoader(trainDataset,
                                              batch_sampler=trainDataset.getSampler(
                                                  batchSize,
                                                  args.samplingType,
                                                  True),
                                              num_workers=args.nGPU)
    valLoader = torch.utils.data.DataLoader(valDataset,
                                            batch_sampler=valDataset.getSampler(
                                                batchSize,
                                                args.samplingType, False),
                                            num_workers=args.nGPU)
    run(trainLoader,
        valLoader,
        cpcModel,
        cpcCriterion,
        args.nEpoch,
        args.pathCheckpoint,
        optimizer,
        logs)

def parseArgs(argv):
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
    parser.add_argument('--nEpoch', type=int, default=200)
    parser.add_argument('--samplingType', type=str, default='uniform',
                        choices=['samespeaker', 'uniform',
                                 'samesequence', 'sequential'])
    parser.add_argument('--nGPU', type=int, default=-1)
    parser.add_argument('--batchSizeGPU', type=int, default=8)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--oneSeq', action='store_true')
    parser.add_argument('--excluding', action='store_true')
    parser.add_argument('--file_extension', type=str, default=".flac")
    parser.add_argument('--dataset_levels', type=int, default=2)

    return parser.parse_args(argv)


if __name__ == "__main__":

    args = parseArgs(sys.argv[1:])
    main(args)
