import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from metric import trainSpeakerSeprarability
from dataset import AudioBatchData, AudioBatchDataset

import numpy as np

import math
import random

import sys

import visdom
vis = visdom.Visdom()


###########################################
# Networks
###########################################

class EncoderNetwork(nn.Module):

    def __init__(self,
                 sizeHidden = 512):

        super(EncoderNetwork, self).__init__()
        self.conv0 = nn.Conv1d(1, sizeHidden, 10, stride=5, padding=3)
        self.batchNorm0 = nn.BatchNorm1d(sizeHidden)
        self.conv1 = nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4, padding=2)
        self.batchNorm1 = nn.BatchNorm1d(sizeHidden)
        self.conv2 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm1d(sizeHidden)
        self.conv3 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm1d(sizeHidden)
        self.conv4 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm1d(sizeHidden)

    def getDimOutput(self):

        return self.conv4.out_channels

    def forward(self, x):

        x = F.relu(self.batchNorm0(self.conv0(x)))
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        x = F.relu(self.batchNorm4(self.conv4(x)))

        return x

class AutoregressiveNetwork(nn.Module):

    def __init__(self,
                 dimEncoded,
                 dimOutput):

        super(AutoregressiveNetwork, self).__init__()

        self.baseNet = nn.GRU(dimEncoded, dimOutput, num_layers =1)

    def getDimOutput(self):
        return self.baseNet.hidden_size

    def forward(self, x):

        return self.baseNet(x)[0]

class PredictionNetwork(nn.Module):

    def __init__(self,
                 nPredicts,
                 dimOutputAR,
                 dimOutputEncoder):

        super(PredictionNetwork, self).__init__()
        self.predictors = nn.ModuleList()

        for i in range(nPredicts):

            self.predictors.append(nn.Linear(dimOutputAR, dimOutputEncoder, bias = False))

    def forward(self, c, candidates):

        assert(len(candidates) == len(self.predictors))

        out = []
        for k in range(len(self.predictors)):

            # torch.nn.Bilinear ? Replace
            locC = self.predictors[k](c)
            locC = locC.view(locC.size(0), 1, locC.size(1), locC.size(2))
            outK = (locC*candidates[k]).mean(dim=3)

            out.append(outK)

        return out

###########################################
# Sampling
###########################################

def getNegativeSamples(encodedData,
                       windowSize,
                       negativeSamplingExt,
                       nGtSequence):

    # Correct the number of negative samples to make sure that the number of
    # indices to draw is lower than the available number of indices
    batchSize = encodedData.size(1)
    negativeSamplingExt = min(negativeSamplingExt, windowSize * (batchSize - nGtSequence))

    # The ground truth data will always be the first item
    labelLoss = torch.zeros((windowSize),
                            dtype = torch.long,
                            device = encodedData.device)

    #Sample on all encoded units
    dimEncoded = encodedData.size(2)
    negativeSamplesExt = encodedData[:windowSize, nGtSequence:, :].contiguous().view(-1, dimEncoded)
    nNegativeExt = negativeSamplesExt.size(0)

    extIdx = np.random.randint(0, nNegativeExt,
                               size=(negativeSamplingExt * windowSize * nGtSequence))

    negExt = negativeSamplesExt[extIdx].view(windowSize,
                                             negativeSamplingExt,
                                             nGtSequence,
                                             dimEncoded)

    return negExt, labelLoss

def getFullSamples(negativeSample,
                   gtPredictions,
                   nPredicts):

    outputs = []
    for k in range(1, nPredicts + 1):

        # Positive samples
        if k < nPredicts:
            posSeq = gtPredictions[k:-(nPredicts-k)]
        else:
            posSeq = gtPredictions[k:]

        posSeq = posSeq.view(posSeq.size(0), 1, posSeq.size(1), posSeq.size(2))

        # Full sequence
        fullSeq = torch.cat((posSeq, negativeSample), dim =1)
        outputs.append(fullSeq)

    return outputs

###########################################
# Metric
###########################################


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
        strLog = [key] + ["{:10.6f}".format(s) for s in logs[key][1:]]
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

def trainStep(dataset,
              batchSize,
              n_devices,
              nPredicts,
              negativeSamplingExt,
              gEncoder,
              gGar,
              wPrediction,
              optimizer,
              lossCriterion,
              logStep):

    gEncoder.train()
    gGar.train()
    wPrediction.train()

    dataLoader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batchSize,
                                             shuffle=True,
                                             num_workers=n_devices)

    logs = {"locLoss": np.zeros(nPredicts + 1),
            "locAcc": np.zeros(nPredicts + 1),
            "locMinMax": np.zeros(nPredicts + 1),
            "step":0}

    for step, fulldata in enumerate(dataLoader):

        optimizer.zero_grad()

        batchData, _ = fulldata

        if batchData.size(0) < batchSize:
            continue

        batchData = batchData.to(gEncoder.output_device)
        encodedData = gEncoder(batchData).permute(2, 0, 1)

        #Data have now the following structure (seq, batchSize, nChannels)
        hiddenEncoder = encodedData.size(2)

        #We are going to perform one prediction sequence per GPU
        gtPredictions = encodedData[:,:n_devices, :].view(-1, n_devices, hiddenEncoder)
        inputGar = encodedData[:-nPredicts, :n_devices, :].view(-1, n_devices, hiddenEncoder)

        cFeature = gGar(inputGar)

        windowSize = inputGar.size(0)

        negExt, labelLoss = getNegativeSamples(encodedData,
                                               windowSize,
                                               negativeSamplingExt,
                                               n_devices)
        fullSeq = getFullSamples(negExt,
                                 gtPredictions,
                                 nPredicts)
        totLoss = 0
        predictions = wPrediction(cFeature, fullSeq)

        for k, locPreds in enumerate(predictions):
            for gtSeq in range(n_devices):
                lossK= lossCriterion(locPreds[:, :, gtSeq], labelLoss)
                totLoss+=lossK
                logs["locLoss"][k + 1] += lossK.item()

                _, predLabel = locPreds[:, :, gtSeq].max(1)
                _, worstLabel = locPreds[:, :, gtSeq].min(1)
                accK = float(torch.sum(predLabel == 0).item()) / (n_devices * windowSize)

                logs["locAcc"][k + 1] += accK
                logs["locMinMax"][k + 1] += float(torch.sum(predLabel == worstLabel).item()) / (n_devices * windowSize)

        totLoss.backward()
        optimizer.step()
        logs["step"]+=1

    updateAndShowLogs("Update %d, training loss:" % (logs["step"] + 1), logs, nPredicts)

    return logs

def valStep(dataset,
            batchSize,
            n_devices,
            nPredicts,
            negativeSamplingExt,
            gEncoder,
            gGar,
            wPrediction,
            lossCriterion):

    gEncoder.eval()
    gGar.eval()
    wPrediction.eval()

    dataLoader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batchSize,
                                             shuffle=True,
                                             num_workers=n_devices)

    locLoss =  np.zeros(nPredicts + 1)
    locAcc = np.zeros(nPredicts + 1)
    locMinMax = np.zeros(nPredicts + 1)

    logs = {"locLoss": np.zeros(nPredicts + 1),
            "locAcc": np.zeros(nPredicts + 1),
            "locMinMax": np.zeros(nPredicts + 1),
            "step":0}

    for step, fulldata in enumerate(dataLoader):

        batchData, _ = fulldata

        if batchData.size(0) < batchSize:
            continue

        batchData = batchData.to(gEncoder.output_device)
        encodedData = gEncoder(batchData).permute(2, 0, 1)

        #Data have now the following structure (seq, batchSize, nChannels)
        hiddenEncoder = encodedData.size(2)

        #We are going to perform one prediction sequence per GPU
        gtPredictions = encodedData[:,:n_devices, :].view(-1, n_devices, hiddenEncoder)
        inputGar = encodedData[:-nPredicts, :n_devices, :].view(-1, n_devices, hiddenEncoder)

        cFeature = gGar(inputGar)

        windowSize = inputGar.size(0)

        negExt, labelLoss = getNegativeSamples(encodedData,
                                               windowSize,
                                               negativeSamplingExt,
                                               n_devices)
        fullSeq = getFullSamples(negExt,
                                 gtPredictions,
                                 nPredicts)
        totLoss = 0
        predictions = wPrediction(cFeature, fullSeq)

        for k, locPreds in enumerate(predictions):
            for gtSeq in range(n_devices):
                lossK= lossCriterion(locPreds[:, :, gtSeq], labelLoss)
                totLoss+=lossK
                logs["locLoss"][k + 1] += lossK.item()

                _, predLabel = locPreds[:, :, gtSeq].max(1)
                _, worstLabel = locPreds[:, :, gtSeq].min(1)
                accK = float(torch.sum(predLabel == 0).item()) / (n_devices * windowSize)

                logs["locAcc"][k + 1] += accK
                logs["locMinMax"][k + 1] += float(torch.sum(predLabel == worstLabel).item()) / (n_devices * windowSize)

    logs["step"] = step

    updateAndShowLogs("Validation loss:", logs, nPredicts)

    return logs

def train(pathDataset,
          hiddenEncoder = 512,
          hiddenGar = 256,
          nPredicts = 12,
          nInputs = 116,
          sizeSample = 160,
          nUpdates = 300000,
          negativeSamplingExt = 128):

    # Build the dataset
    audioData = AudioBatchData(pathDataset)

    # Initialize the networks
    gEncoder = EncoderNetwork(sizeHidden = hiddenEncoder)
    gGar = AutoregressiveNetwork(hiddenEncoder, hiddenGar)
    wPrediction = PredictionNetwork(nPredicts, hiddenGar, hiddenEncoder)

    # Loss criterion
    lossCriterion = nn.CrossEntropyLoss()

    # GPU
    device = torch.device("cuda:0")
    n_devices = 1#torch.cuda.device_count()
    batchSize = 8 * n_devices

    gEncoder = nn.DataParallel(gEncoder)
    gGar = nn.DataParallel(gGar)
    wPrediction = nn.DataParallel(wPrediction)

    gEncoder.to(device)
    gGar.to(device)
    wPrediction.to(device)

    # Optimizer
    g_params = list(gEncoder.parameters()) \
             + list(gGar.parameters()) \
             + list(wPrediction.parameters())

    optimizer = torch.optim.Adam(g_params, lr=2e-4)

    # Estimate the number of epochs to perform
    sizeWindow = sizeSample * (nInputs + nPredicts)
    nWindows = len(audioData) / sizeWindow
    nEpoch = 10

    print("Dataset size: %d , running %d epochs" % (nWindows, nEpoch))

    #  Logs
    logsTrain = {"epoch":[]}
    logsVal = {"epoch": []}
    windowTokenTrain = None
    windowTokenVal = None

    logStep = 1000

    # Train / val split
    #sizeDataset = len(AudioBatchDataset(audioData, offset=sizeWindow, sizeWindow=sizeWindow))
    #indices = torch.randperm(sizeDataset)

    # There are still some overlapps with this configuration
    # TODO: find better
    # Would it be legitimate to split by speaker ? Speaker train / speaker val ?
    #shareTrain = int(0.8*sizeDataset)
    #indicesTrain = indices[:shareTrain]
    #indicesVal = indices[shareTrain:]

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

        locLogs = trainStep(trainDataset, batchSize, n_devices,
                            nPredicts, negativeSamplingExt,
                            gEncoder, gGar, wPrediction, optimizer,
                            lossCriterion, logStep)

        for key, value in locLogs.items():
            if key not in logsTrain:
                logsTrain[key] = [None for x in range(epoch)]
            logsTrain[key].append(value)

        logsTrain["epoch"].append(epoch)
        windowTokenTrain = publishLogs(logsTrain, name = "CPC training", window_tokens = windowTokenTrain)

        locLogs = valStep(valDataset, batchSize, n_devices, nPredicts, negativeSamplingExt,
                gEncoder, gGar, wPrediction, lossCriterion)

        for key, value in locLogs.items():
            if key not in logsVal:
                logsVal[key] = [None for x in range(epoch)]
            logsVal[key].append(value)

        logsVal["epoch"].append(epoch)
        windowTokenVal = publishLogs(logsVal, name = "CPC validation", window_tokens=windowTokenVal)

        #trainSpeakerSeprarability(audioData, gEncoder.module, 5, 8,
        #                          trainEncoder = False)


#The loss profile is indeed strange
# Perform some trivial supervised task to check that everything works
pathDB = '/private/home/mriviere/LibriSpeech/train-clean-100/'
train(pathDB)

hiddenEncoder = 512
gEncoder = EncoderNetwork(sizeHidden = hiddenEncoder)
audioData = AudioBatchData(pathDB)
nEpoch = 10
nSamples = 8
trainSpeakerSeprarability(audioData,
                          gEncoder,
                          nEpoch,
                          nSamples,
                          trainEncoder = False)
