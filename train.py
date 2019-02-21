import os
import torch
import torch.nn as nn
from torch.utils.data import Subset

from dataset import AudioBatchData, AudioBatchDataset
from model import CPCModel
from criterion import CPCUnsupersivedCriterion, SpeakerCriterion

import numpy as np

import math
import random
import argparse
import sys

import visdom
vis = visdom.Visdom()

###########################################
# Main
###########################################

def updateAndShowLogs(text, logs, nPredicts):

    logStep = logs["step"]

    print("")
    print("".join(['-' for x in range(50)]))
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

    print("".join(['-' for x in range(50)]))

def publishLogs(data, name="", window_tokens=None, env="main"):

    if window_tokens is None:
        window_tokens = {key: None for key in data}

    for key, plot in data.items():

        if key in ("step", "epoch"):
            continue

        nItems = len(plot)
        inputY = np.array([plot[x] for x in range(nItems) if 0 is not None])
        inputX = np.array([ data["epoch"][x] for x in range(nItems) if plot[x] is not None])

        opts = {'title': name + " "  + key,
                'legend': [str(x) for x in range(len(plot[0]))], 'xlabel': 'epoch', 'ylabel': 'loss'}

        window_tokens[key] = vis.line(X=inputX, Y=inputY, opts=opts,
                                      win=window_tokens[key], env=env)

    return window_tokens

def trainStep(dataLoader,
              model,
              cpcCriterion,
              optimizer):

    model.train()
    cpcCriterion.train()

    logs = {"step":0}

    for step, fulldata in enumerate(dataLoader):

        optimizer.zero_grad()

        batchData, label = fulldata

        if batchData.size(0) <= cpcCriterion.nGtSequence:
            continue

        batchData = batchData.cuda()
        label = label.cuda()
        cFeature, gtPredictions, otherEncoded = model(batchData,
                                                      nAR = cpcCriterion.nGtSequence)

        allLosses, allAcc = cpcCriterion(cFeature, gtPredictions, otherEncoded, label)

        totLoss = allLosses.sum()
        totLoss.backward()
        optimizer.step()

        if "locLoss_train" not in logs:
            logs["locLoss_train"] = np.zeros(allLosses.size(0))
            logs["locAcc_train"] = np.zeros(allLosses.size(0))

        logs["step"]+=1
        logs["locLoss_train"]+= allLosses.detach().cpu().numpy()
        logs["locAcc_train"] += allAcc.cpu().numpy()

    updateAndShowLogs("Update %d, training loss:" % (logs["step"] + 1), logs, logs["locLoss_train"].shape[0])
    return logs

def valStep(dataLoader,
            model,
            cpcCriterion):

    model.eval()
    cpcCriterion.eval()

    logs = {"step":0}
    for step, fulldata in enumerate(dataLoader):

        batchData, label = fulldata

        if batchData.size(0) <= cpcCriterion.nGtSequence:
            continue

        batchData = batchData.cuda()
        label =label.cuda()
        cFeature, gtPredictions, otherEncoded = model(batchData,
                                                      nAR = cpcCriterion.nGtSequence)

        if otherEncoded.size() ==0:
            print(batchData.size())

        allLosses, allAcc = cpcCriterion(cFeature, gtPredictions, otherEncoded, label)

        if "locLoss_val" not in logs:
            logs["locLoss_val"] = np.zeros(allLosses.size(0))
            logs["locAcc_val"] = np.zeros(allLosses.size(0))

        logs["step"]+=1
        logs["locLoss_val"]+= allLosses.detach().cpu().numpy()
        logs["locAcc_val"] += allAcc.cpu().numpy()

    logs["step"] = step
    updateAndShowLogs("Validation loss:", logs, logs["locLoss_val"].shape[0])

    return logs

def run(trainDataset,
          valDataset,
          cpcModel,
          cpcCriterion,
          optimizeModel,
          nEpoch,
          batchSize,
          learningRate,
          pathCheckpoint):

    cpcCriterion.cuda()
    cpcModel.cuda()

    # Optimizer
    g_params = list(cpcCriterion.parameters())

    if optimizeModel:
        print("Optimizing model")
        g_params+= list(cpcModel.parameters())

    # Nombre magique
    optimizer = torch.optim.Adam(g_params, lr=learningRate)

    print("Dataset size: %d bits, running %d epochs" % (len(audioData), nEpoch))

    #  Logs
    logs = {"epoch":[]}
    windowToken = None

    # Train / val split : by speakers
    nSpeakers = audioData.getNSpeakers()
    speakerTrain = int(0.8 * nSpeakers)
    baseOffsetTrain = audioData.getSpeakerOffset(speakerTrain)

    for epoch in range(nEpoch):

        print("Starting epoch %d" % epoch)

        print("Training dataset %d samples, Validation dataset %d samples" % \
            (len(trainDataset), len(valDataset)))

        trainLoader = torch.utils.data.DataLoader(trainDataset,
                                                  batch_size=batchSize,
                                                  shuffle=True,
                                                  num_workers=2)

        locLogsTrain = trainStep(trainLoader, cpcModel, cpcCriterion, optimizer)

        valLoader = torch.utils.data.DataLoader(valDataset,
                                                batch_size=batchSize,
                                                shuffle=True,
                                                num_workers=2)
        locLogsVal = valStep(valLoader, cpcModel, cpcCriterion)

        for key, value in dict(locLogsTrain, **locLogsVal).items():
            if key not in logs:
                logs[key] = [None for x in range(epoch)]
            logs[key].append(value)

        logs["epoch"].append(epoch)
        windowToken = publishLogs(logs, name = "CPC validation", window_tokens=windowToken)

        # Dirty checkpoint save
        stateDict = {"gEncoder": cpcModel.state_dict(),
                     "cpcCriterion": cpcCriterion.state_dict()}

        #torch.save(stateDict, pathCheckpoint)

if __name__ == "__main__":

    # Run parameters
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--pathDB', type=str, default="/private/home/mriviere/LibriSpeech/train-clean-100/") #TODO: remove in the future
    parser.add_argument('--hiddenEncoder', type=int, default=512)
    parser.add_argument('--hiddenGar', type=int, default=256)
    parser.add_argument('--nPredicts', type=int, default=4)
    parser.add_argument('--negativeSamplingExt', type=int, default=128)
    parser.add_argument('--nGtSequence', type=int, default=1)
    parser.add_argument('--supervised', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--load', type=str, default="")
    parser.add_argument('--learningRate', type=float, default=2e-4)
    parser.add_argument('--pathCheckpoint', type=str, default='checkpoint.pt')
    parser.add_argument('--sizeWindow', type=int, default=20480)
    parser.add_argument('--nEpoch', type=int, default=10)

    args = parser.parse_args()

    audioData = AudioBatchData(args.pathDB)
    cpcModel = CPCModel(args.hiddenEncoder, args.hiddenGar)

    if args.load !="":
        print("Loading checkpoint " + args.load)
        state_dict = torch.load(args.load)
        cpcModel.load_state_dict(state_dict["gEncoder"])

    baseDataset = AudioBatchDataset(audioData, args.sizeWindow)
    sizeDataset = len(baseDataset)
    sizeTrain = int(0.8 * sizeDataset)

    indices = torch.randperm(sizeDataset)
    trainDataset = Subset(baseDataset, indices[:sizeTrain])
    valDataset = Subset(baseDataset, indices[sizeTrain:])

    batchSize = 8 * args.nGtSequence

    if args.supervised:
        cpcCriterion = SpeakerCriterion(args.hiddenGar, audioData.getNSpeakers(), 8)
        indices = torch.randperm(sizeDataset)

    else:
        cpcCriterion = CPCUnsupersivedCriterion(args.nPredicts, args.hiddenGar,
                                                args.hiddenEncoder,
                                                args.negativeSamplingExt,
                                                args.nGtSequence)

    optimizeModel = not args.eval

    if args.eval:
        print("Evaluation mode")

    run(trainDataset,
        valDataset,
        cpcModel,
        cpcCriterion,
        optimizeModel,
        args.nEpoch,
        batchSize,
        args.learningRate,
        args.pathCheckpoint)
