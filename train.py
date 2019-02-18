import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset

import numpy as np

import torchaudio
import math
import random

import sys

###########################################
# Dataset
###########################################

class AudioBatchData:

    # Work on this and on the sampler
    def __init__(self,
                 path):

        self.dbPath = path
        self.loadAll()

    def loadAll(self):

        # Speakers
        self.speakers = [f for f in os.listdir(self.dbPath) \
                         if os.path.isdir(os.path.join(self.dbPath, f))]

        self.speakers =self.speakers[:5]

        # Labels
        self.speakerLabel = [0]
        self.seqLabel = [0]

        # Data
        self.data = []

        itemIndex = 0
        seqIndex = 0
        speakerIndex=0

        for indexSpeaker, speaker in enumerate(self.speakers):
            refPath = os.path.join(self.dbPath, speaker)
            chapters = [ f for f in os.listdir(refPath) \
                        if os.path.isdir(os.path.join(refPath, f))]

            for chapter in chapters:
                chapterPath = os.path.join(refPath, chapter)
                #Debugging only

                for seqName in os.listdir(chapterPath):
                    if os.path.splitext(seqName)[1] != '.flac':
                        continue

                    seqPath = os.path.join(chapterPath, seqName)
                    seq = torchaudio.load(seqPath)[0].view(-1)

                    sizeSeq = seq.size(0)
                    seqIndex+= sizeSeq
                    speakerIndex+= sizeSeq

                    self.data.append(seq)
                    self.seqLabel.append(seqIndex)
                    itemIndex+=1

            self.speakerLabel.append(speakerIndex)

        self.data = torch.cat(self.data, dim = 0)

    def getLabel(self, idx):

       idSpeaker = next(x[0] for x in enumerate(self.speakerLabel) if x[1] >= idx) -1
       return idSpeaker

    def __len__(self):
        return len(self.data)

    def getNSpeakers(self):
        return len(self.speakers)

class AudioBatchDataset(Dataset):

    def __init__(self,
                 batchData,
                 offset =0,
                 sizeWindow=2048,
                 maxOffset = -1):

        self.batchData = batchData
        self.offset = offset
        self.sizeWindow = sizeWindow
        self.maxOffset = maxOffset

        if self.maxOffset <= 0:
            self.maxOffset = len(self.batchData)

    def __len__(self):

        return int(math.floor((self.maxOffset - self.offset) / self.sizeWindow))

    def __getitem__(self, idx):

        windowOffset = self.offset + idx * self.sizeWindow
        speakerLabel = torch.tensor(self.batchData.getLabel(windowOffset), dtype=torch.long)

        return self.batchData.data[windowOffset:(self.sizeWindow + windowOffset)].view(1, -1), speakerLabel

###########################################
# Networks
###########################################

class EncoderNetwork(nn.Module):

    def __init__(self,
                 sizeHidden = 512):

        super(EncoderNetwork, self).__init__()
        self.conv0 = nn.Conv1d(1, sizeHidden, 10, stride=5)
        self.conv1 = nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4)
        self.conv2 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2)
        self.conv3 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2)
        self.conv4 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2)

    def getDimOutput(self):

        return self.conv4.out_channels

    def forward(self, x):

        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

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

            self.predictors.append(nn.Linear(dimOutputAR, dimOutputEncoder))

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

def trainSpeakerSeprarability(audioData,
                              gEncoder,
                              gAR,
                              nEpoch,
                              sizeAudioSample,
                              nSamples,
                              trainEncoder = False):

    nSpeakers = dataset.getNSpeakers()
    print("%d images, %d speakers" % (len(audioData), nSpeakers))

    # Get
    linearSpeakerClassifier = nn.Linear(gEncoder.getDimOutput() * 8, nSpeakers)

    criterion = nn.CrossEntropyLoss()

    batchSize = 16
    n_devices = 2
    device = torch.device("cuda:0")

    lr = 1e-4

    gEncoder = nn.DataParallel(gEncoder)
    gEncoder.to(device)
    gAR = nn.DataParallel(gAR)
    gAR.to(device)
    linearSpeakerClassifier=nn.DataParallel(linearSpeakerClassifier)
    linearSpeakerClassifier.to(device)

    optimizerClassifier = torch.optim.Adam(list(linearSpeakerClassifier.parameters()), lr=lr)
    g_params = list(gEncoder.parameters()) \
             + list(gAR.parameters())

    optimizerG = torch.optim.Adam(g_params, lr=lr)

    schedulerC = torch.optim.lr_scheduler.StepLR(optimizerClassifier, step_size=3, gamma=0.1)
    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=3, gamma=0.1)

    for epoch in range(nEpoch):

        print("Epoch %d" % epoch)

        for isTrain in [True, False]:

            locLoss = 0
            locAcc = 0
            dataLoader = torch.utils.data.DataLoader(datasetMatchTrain[isTrain],
                                                     batch_size=batchSize,
                                                     shuffle=True,
                                                     num_workers=n_devices)
            nStep = 0

            for fullData in dataLoader:

                optimizerClassifier.zero_grad()
                optimizerG.zero_grad()

                batch, labels = fullData
                batch = batch.to(device)
                labels = labels.to(device)

                encodedData = gEncoder(batch)
                encodedData = encodedData.contiguous().view(encodedData.size(0), -1)

                predictedLabels = linearSpeakerClassifier(encodedData)
                labels = labels.view(-1)
                _, predictions = predictedLabels.max(1)

                loss = criterion(predictedLabels, labels)
                loss.backward()

                locLoss+=loss.item()
                locAcc+= float((predictions == labels).sum().item()) / (labels.size(0))

                optimizerClassifier.step()

                if trainEncoder:
                    optimizerG.step()
                nStep+=1

            locLoss /= nStep
            locAcc /= nStep
            if isTrain:
                print("Loss train %f " % locLoss)
                print("Acc train %f" %locAcc)
            else:
                print("Loss val %f " % locLoss)
                print("Acc val %f" %locAcc)

            print("")
            schedulerG.step()
            schedulerG.step()

###########################################
# Main
###########################################

def updateLogs(text, locLoss, locAcc, locMinMax, logStep):

    locLoss /= logStep
    locAcc /= logStep
    locMinMax /= logStep

    nPredicts = len(locLoss)

    strSteps = ['Step'] + [str(s) for s in range(1, nPredicts)]
    strLoss = ['Loss'] + ["{:10.6f}".format(s) for s in locLoss[1:]]
    strAcc = ['Accuracy'] + ["{:10.6f}".format(s) for s in locAcc[1:]]
    strMinMax = ['Flat share'] + ["{:10.6f}".format(s) for s in locMinMax[1:]]
    formatCommand = ' '.join(['{:>16}' for x in range(nPredicts)])

    print("")
    print("".join(['-' for x in range(50)]))
    print(text)
    print(formatCommand.format(*strSteps))
    print(formatCommand.format(*strLoss))
    print(formatCommand.format(*strAcc))
    print(formatCommand.format(*strMinMax))
    print("".join(['-' for x in range(50)]))

    locLoss[:] = 0
    locAcc[:] = 0
    locMinMax[:] = 0

def train(pathDataset,
          hiddenEncoder = 512,
          hiddenGar = 256,
          nPredicts = 4,
          nInputs = 116,
          sizeSample = 160,
          nUpdates = 300000,
          negativeSamplingExt = 128,
          negativeSamplingIn = 1):

    # Build the dataset
    audioData = AudioBatchData(pathDataset)

    # Initialize the networks
    gEncoder = EncoderNetwork(sizeHidden = hiddenEncoder)
    gGar = AutoregressiveNetwork(hiddenEncoder, hiddenGar)
    wPrediction = PredictionNetwork(nPredicts, hiddenGar, hiddenEncoder)

    # Loss criterion
    lossCriterion = nn.CrossEntropyLoss()
    nTotSamples = negativeSamplingExt + negativeSamplingIn + 1

    # GPU
    device = torch.device("cuda:0")
    n_devices = 2#torch.cuda.device_count()
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
    lossLog = []
    accLog = []

    logStep = min(1000, int(math.ceil(nWindows / batchSize)))
    locLoss = {True : np.zeros(nPredicts + 1), False: np.zeros(nPredicts + 1)}
    locAcc = {True : np.zeros(nPredicts + 1), False: np.zeros(nPredicts + 1)}
    locMinMax = {True : np.zeros(nPredicts + 1), False: np.zeros(nPredicts + 1)}

    nUpdate = 0

    for epoch in range(nEpoch):

        print("Starting epoch %d" % epoch)
        offset = random.randint(0, sizeWindow)

        dataset = AudioBatchDataset(audioData,
                                    offset = offset,
                                    sizeWindow = sizeWindow)

        sizeTrain = int(0.8 * len(dataset))
        sizeVal = len(dataset) - sizeTrain
        trainDataset, valDataset = torch.utils.data.random_split(dataset,
                                                                 [sizeTrain,
                                                                    sizeVal])

        isTrainDataset = {True: trainDataset, False: valDataset}

        for isTrain in [True, False]:

            # TODO: faire passer au modele son statut train / val (source de bugs ex dropout)

            dataLoader = torch.utils.data.DataLoader(isTrainDataset[isTrain],
                                                     batch_size=batchSize,
                                                     shuffle=True,
                                                     num_workers=n_devices)

            # Load everything + random offset at each epoch or each batch
            # keep hidden between consecutive batches ?

            for step, fulldata in enumerate(dataLoader):

                optimizer.zero_grad()

                batchData, _ = fulldata

                if batchData.size(0) < batchSize:
                    continue

                batchData = batchData.to(device)
                encodedData = gEncoder(batchData).permute(2, 0, 1)

                #Data have now the following structure (seq, batchSize, nChannels)
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
                        locLoss[isTrain][k] += lossK.item()

                        _, predLabel = locPreds[:, :, gtSeq].max(1)
                        _, worstLabel = locPreds[:, :, gtSeq].min(1)
                        accK = float(torch.sum(predLabel == 0).item()) / (n_devices * windowSize)

                        locAcc[isTrain][k] += accK
                        locMinMax[isTrain][k] += float(torch.sum(predLabel == worstLabel).item()) / (n_devices * windowSize)

                if isTrain:
                    totLoss.backward()
                    optimizer.step()
                    nUpdate+=1

                    if nUpdate % logStep == logStep -1:
                        updateLogs("Update %d, training loss:" % (nUpdate + 1), locLoss[isTrain], locAcc[isTrain], locMinMax[isTrain], logStep)

            if not isTrain:
                updateLogs("Validation loss:", locLoss[isTrain], locAcc[isTrain], locMinMax[isTrain], step)

#The loss profile is indeed strange
# Perform some trivial supervised task to check that everything works


pathDB = '/private/home/mriviere/LibriSpeech/train-clean-100/'
#audioData = AudioBatchData(pathDB)
#testDB = AudioBatchDataset(pathDB)
train(pathDB)

#audioDataset = AudioBatchData(pathDB)
#print(audioDataset.getLabel(48355924))
