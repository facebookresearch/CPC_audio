import torch
import torch.nn as nn
import numpy as np


class PredictionNetwork(nn.Module):

    def __init__(self,
                 nPredicts,
                 dimOutputAR,
                 dimOutputEncoder):

        super(PredictionNetwork, self).__init__()
        self.predictors = nn.ModuleList()

        for i in range(nPredicts):
            self.predictors.append(
                nn.Linear(dimOutputAR, dimOutputEncoder, bias=False))

    def forward(self, c, candidates):

        assert(len(candidates) == len(self.predictors))

        out = []
        for k in range(len(self.predictors)):

            locC = self.predictors[k](c)
            locC = locC.view(locC.size(0), 1, locC.size(1), locC.size(2))
            outK = (locC*candidates[k]).mean(dim=3)
            out.append(outK)
        return out


class CPCUnsupersivedCriterion(nn.Module):

    def __init__(self,
                 nPredicts,             # Number of steps
                 dimOutputAR,           # Dimension of G_ar
                 dimOutputEncoder,      # Dimension of the convolutional net
                 negativeSamplingExt,   # Number of negative samples to draw
                 mode=None):

        super(CPCUnsupersivedCriterion, self).__init__()
        self.wPrediction = PredictionNetwork(
            nPredicts, dimOutputAR, dimOutputEncoder)
        self.nPredicts = nPredicts
        self.negativeSamplingExt = negativeSamplingExt
        self.lossCriterion = nn.CrossEntropyLoss()

        if mode not in [None, "reverse", "cloze"]:
            raise ValueError("Invalid mode")

        self.mode = mode
        if mode == "cloze" and dimOutputAR % 2 != 0:
            raise ValueError("On cloze mode dimOutputAR should be an even\
                              number")

    def sample(self, encodedData, windowSize):

        batchSize, nNegativeExt, dimEncoded = encodedData.size()
        outputs = []

        negExt = encodedData.contiguous().view(-1, dimEncoded)
        # Draw nNegativeExt * batchSize negative samples anywhere in the batch
        extIdx = np.random.randint(0, nNegativeExt * batchSize,
                                   size=(self.negativeSamplingExt
                                         * windowSize * batchSize))
        negExt = negExt[extIdx].view(batchSize, self.negativeSamplingExt,
                                     windowSize, dimEncoded)

        labelLoss = torch.zeros((batchSize * windowSize),
                                dtype=torch.long,
                                device=encodedData.device)

        for k in range(1, self.nPredicts + 1):

            # Positive samples
            if k < self.nPredicts:
                posSeq = encodedData[:, k:-(self.nPredicts-k)]
            else:
                posSeq = encodedData[:, k:]

            if self.mode == "cloze":
                posSeq = posSeq[:, :-1]

            posSeq = posSeq.view(batchSize, 1, posSeq.size(1), dimEncoded)
            fullSeq = torch.cat((posSeq, negExt), dim=1)
            outputs.append(fullSeq)

        return outputs, labelLoss

    def forward(self, cFeature, encodedData, *args):

        if self.mode == "reverse":
            encodedData = torch.flip(encodedData, [1])
            cFeature = torch.flip(cFeature, [1])

        batchSize, seqSize, dimAR = cFeature.size()
        windowSize = seqSize - self.nPredicts

        if self.mode == "cloze":
            halfSize = dimAR // 2
            windowSize -= 1
            featureStraight = cFeature[:, :windowSize, :halfSize]
            featureReverse = cFeature[:, self.nPredicts:-1, halfSize:]
            cFeature = torch.cat([featureStraight, featureReverse], dim=2)
        else:
            cFeature = cFeature[:, :windowSize]

        sampledData, labelLoss = self.sample(encodedData, windowSize)
        predictions = self.wPrediction(cFeature, sampledData)

        outLosses = [0 for x in range(self.nPredicts)]
        outAcc = [0 for x in range(self.nPredicts)]

        for k, locPreds in enumerate(predictions):
            locPreds = locPreds.permute(0, 2, 1)
            locPreds = locPreds.contiguous().view(-1, locPreds.size(2))
            lossK = self.lossCriterion(locPreds, labelLoss)
            outLosses[k] += lossK.view(1, -1)
            _, predsIndex = locPreds.max(1)
            outAcc[k] += torch.sum(predsIndex == labelLoss).float().view(1, -1)

        return torch.cat(outLosses, dim=1), torch.cat(outAcc, dim=1) / (windowSize * batchSize)


class CPCBertCriterion(nn.Module):

    def __init__(self,
                 dimOutputAR,           # Dimension of G_ar
                 dimOutputEncoder,      # Dimension of the convolutional net
                 negativeSamplingExt):  # Number of negative samples to draw

        super(CPCBertCriterion, self).__init__()
        self.wPrediction = torch.nn.Linear(dimOutputAR,
                                           dimOutputEncoder,
                                           bias=False)

        self.negativeSamplingExt = negativeSamplingExt
        self.lossCriterion = nn.CrossEntropyLoss()

    def sample(self, encodedData, mask):

        batchSize, nNegativeExt, dimEncoded = encodedData.size()
        negExt = encodedData[1 - mask].view(-1, dimEncoded)
        nPos = mask.sum()
        extIdx = np.random.randint(0, negExt.size(0),
                                   size=(self.negativeSamplingExt * nPos))
        negExt = negExt[extIdx].view(nPos, self.negativeSamplingExt,
                                     dimEncoded)

        posData = encodedData[mask].view(nPos, 1, dimEncoded)
        labelLoss = torch.zeros(nPos, dtype=torch.long,
                                device=encodedData.device)

        return torch.cat([posData, negExt], dim=1), labelLoss

    def forward(self, cFeature, encodedData, label):

        batchSize, seqSize, dimAR = cFeature.size()
        samples, labelLoss = self.sample(encodedData, label)
        nPos = labelLoss.size(0)
        predictions = self.wPrediction(cFeature[label]).view(nPos, 1, -1)
        predictions = (samples * predictions).mean(dim=2)
        loss = self.lossCriterion(predictions, labelLoss)
        _, predsIndex = predictions.max(1)
        acc = torch.sum(predsIndex == labelLoss).double(
        ).view(1, -1) / nPos

        return loss.view(1, 1), acc


class SpeakerCriterion(nn.Module):

    def __init__(self, dimEncoder, nSpeakers):

        super(SpeakerCriterion, self).__init__()
        self.linearSpeakerClassifier = nn.Linear(
            dimEncoder, nSpeakers)
        self.lossCriterion = nn.CrossEntropyLoss()
        self.entropyCriterion = nn.LogSoftmax(dim=1)

    def forward(self, cFeature, otherEncoded, label):

        # cFeature.size() : batchSize x seq Size x hidden size
        batchSize = cFeature.size(0)
        cFeature = cFeature[:, -1, :]
        cFeature = cFeature.view(batchSize, -1)
        predictions = self.linearSpeakerClassifier(cFeature)

        if label is None:
            loss = self.entropyCriterion(predictions).mean(dim=1).view(-1)
            acc = torch.zeros(1, 1).cuda()
        else:
            loss = self.lossCriterion(predictions, label).view(1, -1)
            acc = (predictions.max(1)[1] == label).double().mean().view(1, -1)

        return loss, acc


class PhoneCriterion(nn.Module):

    def __init__(self, dimEncoder, nPhones):

        super(PhoneCriterion, self).__init__()
        self.PhoneCriterionClassifier = nn.Linear(dimEncoder, nPhones)
        self.lossCriterion = nn.CrossEntropyLoss()

    def forward(self, cFeature, otherEncoded, label):

        # cFeature.size() : batchSize x seq Size x hidden size
        predictions = self.getPrediction(cFeature)
        label = label.view(-1)
        loss = self.lossCriterion(predictions, label).view(1, -1)
        acc = (predictions.max(1)[1] == label).double().mean().view(1, -1)
        return loss, acc

    def getPrediction(self, cFeature):
        batchSize, seqSize = cFeature.size(0), cFeature.size(1)
        cFeature = cFeature.contiguous().view(batchSize * seqSize, -1)
        return self.PhoneCriterionClassifier(cFeature)


class ModelCriterionCombined(torch.nn.Module):
    def __init__(self, model, criterion):
        super(ModelCriterionCombined, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, data, label):
        c_feature, encoded_data, label = self.model(data, label)
        loss, acc = self.criterion(c_feature, encoded_data, label)
        return loss, acc
