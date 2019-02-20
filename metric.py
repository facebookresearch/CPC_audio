import torch
import torch.nn as nn
from torch.utils.data import Subset

import random
import math

from dataset import AudioBatchDataset

import sys

#TODO: correct. With the BatchNorm involved we need to use the train() / eval()
# modes
def trainSpeakerSeprarability(audioData,
                              gEncoder,
                              nEpoch,
                              nSamples,
                              trainEncoder = False):

    nSpeakers = audioData.getNSpeakers()
    print("".join(['-' for x in range(50)]))
    print("Speaker separability")
    print("%d audio bits, %d speakers" % (len(audioData), nSpeakers))

    # Get
    linearSpeakerClassifier = nn.Linear(gEncoder.getDimOutput() * nSamples, nSpeakers)
    criterion = nn.CrossEntropyLoss()

    # Dataset parameters
    sizeWindow = 160 * nSamples
    batchSize = 16
    device = torch.device("cuda:0")

    lr = 2e-4

    # Optimization
    gEncoder = nn.DataParallel(gEncoder)
    gEncoder.to(device)
    linearSpeakerClassifier=nn.DataParallel(linearSpeakerClassifier)
    linearSpeakerClassifier.to(device)

    optimizerClassifier = torch.optim.Adam(list(linearSpeakerClassifier.parameters()), lr=lr)
    schedulerC = torch.optim.lr_scheduler.StepLR(optimizerClassifier, step_size=3, gamma=0.1)

    if trainEncoder:
        g_params = list(gEncoder.parameters())
        optimizerG = torch.optim.Adam(g_params, lr=lr)
        schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=3, gamma=0.1)

    offset = random.randint(0, sizeWindow)
    dataset = AudioBatchDataset(audioData,
                                offset = offset,
                                sizeWindow = sizeWindow)

    sizeDataset = len(dataset)
    sizeTrain = int(0.8 * sizeDataset)
    indices = torch.randperm(sizeDataset)

    trainDataset = Subset(dataset, indices[:sizeTrain])
    valDataset = Subset(dataset, indices[sizeTrain:])

    for epoch in range(nEpoch):

        print("Epoch %d" % epoch)
        dataLoaderTrain = torch.utils.data.DataLoader(trainDataset,
                                                      batch_size=batchSize,
                                                      shuffle=True,
                                                      num_workers=torch.cuda.device_count())

        gEncoder.train()
        linearSpeakerClassifier.train()
        locLoss=0
        locAcc =0
        nStep=0

        for fullData in dataLoaderTrain:

            optimizerClassifier.zero_grad()
            gEncoder.zero_grad()

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
        print("Loss train %f " % locLoss)
        print("Acc train %f" %locAcc)

        if trainEncoder:
            schedulerG.step()
        schedulerC.step()

        gEncoder.eval()
        linearSpeakerClassifier.eval()
        locLoss=0
        locAcc =0
        nStep=0

        dataLoaderVal = torch.utils.data.DataLoader(valDataset,
                                                    batch_size=batchSize,
                                                    shuffle=True,
                                                    num_workers=torch.cuda.device_count())

        for fullData in dataLoaderVal:

            batch, labels = fullData
            batch = batch.to(device)
            labels = labels.to(device)

            encodedData = gEncoder(batch)
            encodedData = encodedData.contiguous().view(encodedData.size(0), -1)

            predictedLabels = linearSpeakerClassifier(encodedData)
            labels = labels.view(-1)
            _, predictions = predictedLabels.max(1)

            loss = criterion(predictedLabels, labels)

            locLoss+=loss.item()
            locAcc+= float((predictions == labels).sum().item()) / (labels.size(0))
            nStep+=1

        locLoss /= nStep
        locAcc /= nStep
        print("Loss val %f " % locLoss)
        print("Acc val %f" %locAcc)

    print("".join(['-' for x in range(50)]))
    print("")
