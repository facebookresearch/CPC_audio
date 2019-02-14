import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset

import numpy as np

import torchaudio
import math

class AudioBatchDataset(Dataset):

    def __init__(self,
                 path,
                 nSamples = 128, # is that where the 20480 comes from ?
                 sizeSample = 160):

        self.dbPath = path
        self.nSamples = nSamples
        self.sizeSample = sizeSample
        self.sizeWindow = nSamples * sizeSample

        self.prepareDataset()

    def prepareDataset(self):

        # Speakers
        self.speakers = [f for f in os.listdir(self.dbPath) \
                         if os.path.isdir(os.path.join(self.dbPath, f))]

        self.nSpeakers = len(self.speakers)

        # Sequences
        self.sequencesPath = []

        # Data
        self.data = []

        # Speakers labelling (for specific training)
        totLenght = 0
        item = 0
        for indexSpeaker, speaker in enumerate(self.speakers):
            refPath = os.path.join(self.dbPath, speaker)
            chapters = [ f for f in os.listdir(refPath) \
                        if os.path.isdir(os.path.join(refPath, f))]
            for chapter in chapters:
                chapterPath = os.path.join(refPath, chapter)

                for seqName in os.listdir(chapterPath):
                    if os.path.splitext(seqName)[1] != '.flac':
                        continue

                    seqPath = os.path.join(chapterPath, seqName)
                    self.sequencesPath.append(seqPath)
                    data, _ = torchaudio.info(seqPath)
                    lenght = int(data.length / self.sizeWindow)
                    totLenght+= lenght

                    self.data = self.data + [(item,
                                              i*self.sizeWindow)
                                             for i in range(lenght)]
                    item+=1

    def __getitem__(self, idx):

        indexPath, shift = self.data[idx]
        path = self.sequencesPath[indexPath]

        outputAudio, _ = torchaudio.load(path,
                                         num_frames = self.sizeWindow,
                                         offset= shift)

        return outputAudio

    def __len__(self):
        return len(self.data)

class EncoderNetwork(nn.Module):

    def __init__(self,
                 sizeHidden = 512):

        super(EncoderNetwork, self).__init__()
        self.conv0 = nn.Conv1d(1, sizeHidden, 10, stride=5)
        self.conv1 = nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4)
        self.conv2 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2)
        self.conv3 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2)
        self.conv4 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2)

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

    def forward(self, x):

        return self.baseNet(x)[0]

class PredictionNetwork(nn.Module):

    def __init__(self,
                 nPredicts,
                 dimPrediction,
                 dimSample):

        super(PredictionNetwork, self).__init__()

        self.predictors = nn.ModuleList()
        self.nPredicts = nPredicts

        for i in range(self.nPredicts):

            self.predictors.append(nn.Linear(dimPrediction, dimSample))

    def forward(self, c, z, k):

        c = self.predictors[k](c)
        c = c.view(c.size(0), 1, c.size(1), c.size(2))

        out = (c*z).mean(dim=3)
        return out

def train(pathDataset,
          hiddenEncoder = 512,
          hiddenGar = 256,
          nPredicts = 12,
          nInputs = 116,
          sizeSample = 160,
          nUpdates = 3000,
          negativeSamplingExt = 128,
          negativeSamplingIn = 128):

    # Build the dataset
    audioDataset = AudioBatchDataset(pathDataset,
                                     nSamples = nPredicts + nInputs,
                                     sizeSample = sizeSample)

    # Initialize the networks
    gEncoder = EncoderNetwork(sizeHidden = hiddenEncoder)
    gGar = AutoregressiveNetwork(hiddenEncoder, hiddenGar)
    wPrediction = PredictionNetwork(nPredicts, hiddenGar, hiddenEncoder)

    # Loss criterion
    lossCriterion = nn.CrossEntropyLoss()
    nTotSamples = negativeSamplingExt + negativeSamplingIn + 1

    # Estimate the number of epochs to perform
    nWindows = len(audioDataset)
    nEpoch = int(math.ceil(nWindows / nUpdates))

    print("Dataset size: %d , running %d epochs" % (nWindows, nEpoch))

    # GPU
    device = torch.device("cuda:0")
    n_devices = torch.cuda.device_count()
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

    #  Logs
    lossLog = []
    accLog = []

    logStep = 100
    locLoss = np.zeros(nPredicts + 1)
    locAcc = np.zeros(nPredicts + 1)

    for epoch in range(nEpoch):

        print("Starting epoch %d" % epoch)

        dataLoader = torch.utils.data.DataLoader(audioDataset,
                                                 batch_size=batchSize,
                                                 shuffle=True,
                                                 num_workers=n_devices)

        for step, batchData in enumerate(dataLoader):

            optimizer.zero_grad()

            batchData = batchData.to(device)
            encodedData = gEncoder(batchData).permute(2, 0, 1)

            #Data have now the following structure (seq, batchSize, nChannels)
            gtPredictions = encodedData[:,:n_devices, :].view(-1, n_devices, hiddenEncoder)
            inputGar = encodedData[:-nPredicts, :n_devices, :].view(-1, n_devices, hiddenEncoder)

            cFeature = gGar(inputGar)

            #Negative samples
            negativeSamplesExt = encodedData[:, n_devices:, :].contiguous().view(-1, hiddenEncoder)
            nNegativeExt = negativeSamplesExt.size(0)

            windowSize = inputGar.size(0)
            labelLoss = torch.zeros((windowSize * n_devices),
                                    dtype = torch.long,
                                    device = device)

            for k in range(1, nPredicts + 1):

                extIdx = np.random.randint(0, nNegativeExt,
                                           size=(negativeSamplingExt * windowSize * n_devices))

                negExt = negativeSamplesExt[extIdx].view(windowSize,
                                                         negativeSamplingExt,
                                                         n_devices,
                                                         hiddenEncoder)

                # We are quite unlinkely to pick the good samples randomly from
                # the sequence
                seqNegIdx = np.random.randint(0, windowSize, size=(negativeSamplingIn* windowSize))

                negSeq = inputGar[seqNegIdx].view(windowSize, negativeSamplingIn, n_devices, hiddenEncoder)

                # Positive samples
                if k < nPredicts:
                    posSeq = gtPredictions[k:-(nPredicts-k)]
                else:
                    posSeq = gtPredictions[k:]

                posSeq = posSeq.view(windowSize, 1, n_devices, hiddenEncoder)

                # Full sequence
                fullSeq = torch.cat((posSeq, negSeq, negExt), dim =1)

                # Predictions
                predictions = wPrediction(cFeature, fullSeq, 0)

                # Loss !
                predictions = predictions.permute(0, 2, 1).contiguous().view(n_devices * windowSize, nTotSamples)
                lossK = lossCriterion(predictions, labelLoss)
                lossK.backward(retain_graph = k < nPredicts)

                # Accuracy (train)
                _, predLabel = predictions.max(1)
                accK = float(torch.sum(predLabel == 0).item()) / (n_devices * windowSize)

                locLoss[k] += lossK.item()
                locAcc[k] += accK

            optimizer.step()

            if step % logStep == (logStep -1):

                locLoss /= logStep
                locAcc /= logStep

                strSteps = ['Step'] + [str(s) for s in range(1, nPredicts + 1)]
                strLoss = ['Loss'] + ["{:10.6f}".format(s) for s in locLoss[1:]]
                strAcc = ['Accuracy'] + ["{:10.6f}".format(s) for s in locAcc[1:]]
                formatCommand = ' '.join(['{:>16}' for x in range(nPredicts + 1)])

                print("")
                print("".join(['-' for x in range(50)]))
                print("Update %d :" % (step+1))
                print(formatCommand.format(*strSteps))
                print(formatCommand.format(*strLoss))
                print(formatCommand.format(*strAcc))
                print("".join(['-' for x in range(50)]))

                locLoss[:] = 0
                locAcc[:] = 0


pathDB = '/private/home/mriviere/LibriSpeech/train-clean-100/'
train(pathDB)

#testDb = AudioBatchDataset('/private/home/mriviere/LibriSpeech/train-clean-100/')

#inputWave = testDb[12].view(1, 1, -1)

#testEncoder = EncoderNetwork()
#testEncoded = testEncoder(inputWave)

#testEncoded = testEncoded.permute(2, 0, 1)

#testGar = AutoregressiveNetwork(512, 256)
#testC = testGar(testEncoded)

#testPrediction = PredictionNetwork(12, 256, 512)

#c0 = testC[0, :]
#z24 = testEncoded[24, :]

#print(testPrediction(c0, z24, 2))
