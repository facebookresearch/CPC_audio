import os
import torch
import torch.nn as nn

from dataset import AudioBatchData, AudioBatchDataset
from model import CPCModel
from criterion import CPCUnsupersivedCriterion

import numpy as np

import math
import random

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

        batchData = batchData.to(model.output_device)
        cFeature, gtPredictions, otherEncoded = model(batchData,
                                                      nAR = cpcCriterion.nGtSequence)

        allLosses, allAcc = cpcCriterion.getPredictions(cFeature, gtPredictions, otherEncoded)

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

        batchData = batchData.to(model.output_device)
        cFeature, gtPredictions, otherEncoded = model(batchData,
                                                      nAR = cpcCriterion.nGtSequence)

        if otherEncoded.size() ==0:
            print(batchData.size())

        allLosses, allAcc = cpcCriterion.getPredictions(cFeature, gtPredictions, otherEncoded, label)

        if "locLoss_val" not in logs:
            logs["locLoss_val"] = np.zeros(allLosses.size(0))
            logs["locAcc_val"] = np.zeros(allLosses.size(0))

        logs["step"]+=1
        logs["locLoss_val"]+= allLosses.detach().cpu().numpy()
        logs["locAcc_val"] += allAcc.cpu().numpy()

    logs["step"] = step
    updateAndShowLogs("Validation loss:", logs, logs["locLoss_val"].shape[0])

    return logs

def train(pathDataset,
          hiddenEncoder = 512,
          hiddenGar = 256,
          nPredicts = 4,
          nInputs = 116,
          sizeSample = 160,
          nUpdates = 300000,
          negativeSamplingExt = 128,
          nEpoch=10,
          learningRate=2e-4,
          pathCheckpoint = 'checkpoint.pt',
          nGtSequence=1):

    # Build the dataset
    audioData = AudioBatchData(pathDataset)

    # Initialize the networks
    cpcModel = CPCModel(hiddenEncoder, hiddenGar)
    cpcCriterion = CPCUnsupersivedCriterion(nPredicts, hiddenGar,
                                            hiddenEncoder,
                                            negativeSamplingExt,
                                            nGtSequence)

    # Loss criterion
    lossCriterion = nn.CrossEntropyLoss()

    # GPU
    device = torch.device("cuda:0")
    batchSize = 8 * nGtSequence

    cpcModel = nn.DataParallel(cpcModel)
    cpcModel.to(device)

    # Optimizer
    g_params = list(cpcModel.parameters()) \
             + list(cpcCriterion.parameters())

    # Nombre magique
    optimizer = torch.optim.Adam(g_params, lr=learningRate)

    # Estimate the number of epochs to perform
    sizeWindow = sizeSample * (nInputs + nPredicts)
    nWindows = len(audioData) / sizeWindow

    print("Dataset size: %d , running %d epochs" % (nWindows, nEpoch))

    #  Logs
    logs = {"epoch":[]}
    windowToken = None

    # Train / val split : by speakers
    nSpeakers = audioData.getNSpeakers()
    speakerTrain = int(0.8 * nSpeakers)
    baseOffsetTrain = audioData.getSpeakerOffset(speakerTrain)

    for epoch in range(nEpoch):

        print("Starting epoch %d" % epoch)
        offset = random.randint(0, sizeWindow / 2)

        trainDataset = AudioBatchDataset(audioData,
                                         offset=offset,
                                         sizeWindow=sizeWindow,
                                         maxOffset=baseOffsetTrain + offset)

        valDataset = AudioBatchDataset(audioData,
                                       offset=baseOffsetTrain + offset,
                                       sizeWindow=sizeWindow)

        print("Training dataset %d samples, Validation dataset %d samples" % \
            (len(trainDataset), len(valDataset)))

        trainLoader = torch.utils.data.DataLoader(trainDataset,
                                                 batch_size=batchSize,
                                                 shuffle=True,
                                                 num_workers=nGtSequence)

        locLogsTrain = trainStep(trainLoader, cpcModel, cpcCriterion, optimizer)

        valLoader = torch.utils.data.DataLoader(valDataset,
                                                 batch_size=batchSize,
                                                 shuffle=True,
                                                 num_workers=nGtSequence)
        locLogsVal = valStep(valLoader, cpcModel, cpcCriterion)

        for key, value in dict(locLogsTrain, **locLogsVal).items():
            if key not in logs:
                logs[key] = [None for x in range(epoch)]
            logs[key].append(value)

        logs["epoch"].append(epoch)
        windowToken = publishLogs(logs, name = "CPC validation", window_tokens=windowToken)

        # Dirty checkpoint save
        stateDict = {"gEncoder": cpcModel.module.state_dict(),
                     "cpcCriterion": cpcCriterion.state_dict()}

        torch.save(stateDict, pathCheckpoint)

    return gEncoder.module, gGar.module, wPrediction.module


#The loss profile is indeed strange
# Perform some trivial supervised task to check that everything works
pathDB = '/private/home/mriviere/LibriSpeech/train-clean-100/'
gEncoder, gAR, wPrediction = train(pathDB)


#TODO:
# - moins de lignes ! Bug prone code
# - separer le train en 4 modules: dataloder -> model -> criterion -> metric
# le but est de pouvoir faire passer la boucle de unsupervised a supervised facilement

# A priori mettre la metrique dans un fichier separer -> eval.py
# ou
