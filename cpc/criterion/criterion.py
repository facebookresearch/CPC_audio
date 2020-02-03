# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
from .seq_alignment import collapseLabelChain
from .custom_layers import EqualizedLinear, EqualizedConv1d


class FFNetwork(nn.Module):
    def __init__(self, din, dout, dff, dropout):
        super(FFNetwork, self).__init__()
        self.lin1 = EqualizedLinear(din, dff, bias=True, equalized=True)
        self.lin2 = EqualizedLinear(dff, dout, bias=True, equalized=True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.lin2(self.drop(self.relu(self.lin1(x))))


class ShiftedConv(nn.Module):
    def __init__(self, dimOutputAR, dimOutputEncoder, kernelSize):
        super(ShiftedConv, self).__init__()
        self.module = EqualizedConv1d(dimOutputAR, dimOutputEncoder,
                                      kernelSize, equalized=True,
                                      padding=0)
        self.kernelSize = kernelSize

    def forward(self, x):

        # Input format: N, S, C -> need to move to N, C, S
        N, S, C = x.size()
        x = x.permute(0, 2, 1)

        padding = torch.zeros(N, C, self.kernelSize - 1, device=x.device)
        x = torch.cat([padding, x], dim=2)
        x = self.module(x)
        x = x.permute(0, 2, 1)
        return x


class PredictionNetwork(nn.Module):

    def __init__(self,
                 nPredicts,
                 dimOutputAR,
                 dimOutputEncoder,
                 rnnMode=None,
                 dropout=False,
                 sizeInputSeq=116):

        super(PredictionNetwork, self).__init__()
        self.predictors = nn.ModuleList()
        self.RESIDUAL_STD = 0.01
        self.dimOutputAR = dimOutputAR

        self.dropout = nn.Dropout(p=0.5) if dropout else None
        for i in range(nPredicts):
            if rnnMode == 'RNN':
                self.predictors.append(
                    nn.RNN(dimOutputAR, dimOutputEncoder))
                self.predictors[-1].flatten_parameters()
            elif rnnMode == 'LSTM':
                self.predictors.append(
                    nn.LSTM(dimOutputAR, dimOutputEncoder, batch_first=True))
                self.predictors[-1].flatten_parameters()
            elif rnnMode == 'ffd':
                self.predictors.append(
                    FFNetwork(dimOutputAR, dimOutputEncoder,
                              dimOutputEncoder, 0))
            elif rnnMode == 'conv4':
                self.predictors.append(
                    ShiftedConv(dimOutputAR, dimOutputEncoder, 4))
            elif rnnMode == 'conv8':
                self.predictors.append(
                    ShiftedConv(dimOutputAR, dimOutputEncoder, 8))
            elif rnnMode == 'conv12':
                self.predictors.append(
                    ShiftedConv(dimOutputAR, dimOutputEncoder, 12))
            elif rnnMode == 'transformer':
                from transformers import buildTransformerAR
                self.predictors.append(
                    buildTransformerAR(dimOutputEncoder,
                                       1,
                                       sizeInputSeq,
                                       False))
            else:
                self.predictors.append(
                    nn.Linear(dimOutputAR, dimOutputEncoder, bias=False))
                if dimOutputEncoder > dimOutputAR:
                    residual = dimOutputEncoder - dimOutputAR
                    self.predictors[-1].weight.data.copy_(torch.cat([torch.randn(
                        dimOutputAR, dimOutputAR), self.RESIDUAL_STD * torch.randn(residual, dimOutputAR)], dim=0))

    def forward(self, c, candidates):

        assert(len(candidates) == len(self.predictors))
        out = []

        # UGLY
        if isinstance(self.predictors[0], EqualizedConv1d):
            c = c.permute(0, 2, 1)

        for k in range(len(self.predictors)):

            locC = self.predictors[k](c)
            if isinstance(locC, tuple):
                locC = locC[0]
            if isinstance(self.predictors[k], EqualizedConv1d):
                locC = locC.permute(0, 2, 1)
            if self.dropout is not None:
                locC = self.dropout(locC)
            locC = locC.view(locC.size(0), 1, locC.size(1), locC.size(2))
            outK = (locC*candidates[k]).mean(dim=3)
            out.append(outK)
        return out


class BaseCriterion(nn.Module):

    def warmUp(self):
        return False

    def update(self):
        return


class NoneCriterion(BaseCriterion):
    def __init__(self):
        super(NoneCriterion, self).__init__()

    def forward(self, cFeature, encodedData, label):
        return torch.zeros(1, 1, device=cFeature.device), \
            torch.zeros(1, 1, device=cFeature.device)


class CPCUnsupersivedCriterion(BaseCriterion):

    def __init__(self,
                 nPredicts,             # Number of steps
                 dimOutputAR,           # Dimension of G_ar
                 dimOutputEncoder,      # Dimension of the convolutional net
                 negativeSamplingExt,   # Number of negative samples to draw
                 mode=None,
                 rnnMode=False,
                 dropout=False,
                 speakerEmbedding=0,
                 nSpeakers=0,
                 sizeInputSeq=128):

        super(CPCUnsupersivedCriterion, self).__init__()
        if speakerEmbedding > 0:
            print(
                f"Using {speakerEmbedding} speaker embeddings for {nSpeakers} speakers")
            self.speakerEmb = torch.nn.Embedding(nSpeakers, speakerEmbedding)
            dimOutputAR += speakerEmbedding
        else:
            self.speakerEmb = None

        self.wPrediction = PredictionNetwork(
            nPredicts, dimOutputAR, dimOutputEncoder, rnnMode=rnnMode,
            dropout=dropout, sizeInputSeq=sizeInputSeq - nPredicts)
        self.nPredicts = nPredicts
        self.negativeSamplingExt = negativeSamplingExt
        self.lossCriterion = nn.CrossEntropyLoss()

        if mode not in [None, "reverse"]:
            raise ValueError("Invalid mode")

        self.mode = mode

    def sampleClean(self, encodedData, windowSize):

        batchSize, nNegativeExt, dimEncoded = encodedData.size()
        outputs = []

        negExt = encodedData.contiguous().view(-1, dimEncoded)
        # Draw nNegativeExt * batchSize negative samples anywhere in the batch
        batchIdx = torch.randint(low=0, high=batchSize,
                                 size=(self.negativeSamplingExt
                                       * windowSize * batchSize, ),
                                 device=encodedData.device)

        seqIdx = torch.randint(low=1, high=nNegativeExt,
                               size=(self.negativeSamplingExt
                                     * windowSize * batchSize, ),
                               device=encodedData.device)

        baseIdx = torch.arange(0, windowSize, device=encodedData.device)
        baseIdx = baseIdx.view(1, 1,
                               windowSize).expand(1,
                                                  self.negativeSamplingExt,
                                                  windowSize).expand(batchSize, self.negativeSamplingExt, windowSize)
        seqIdx += baseIdx.contiguous().view(-1)
        seqIdx = torch.remainder(seqIdx, nNegativeExt)

        extIdx = seqIdx + batchIdx * nNegativeExt
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

            posSeq = posSeq.view(batchSize, 1, posSeq.size(1), dimEncoded)
            fullSeq = torch.cat((posSeq, negExt), dim=1)
            outputs.append(fullSeq)

        return outputs, labelLoss

    def getInnerLoss(self):

        return "orthoLoss", self.orthoLoss * self.wPrediction.orthoCriterion()

    def forward(self, cFeature, encodedData, label):

        if self.mode == "reverse":
            encodedData = torch.flip(encodedData, [1])
            cFeature = torch.flip(cFeature, [1])

        batchSize, seqSize, dimAR = cFeature.size()
        windowSize = seqSize - self.nPredicts

        cFeature = cFeature[:, :windowSize]

        sampledData, labelLoss = self.sampleClean(encodedData, windowSize)

        if self.speakerEmb is not None:
            l_ = label.view(batchSize, 1).expand(batchSize, windowSize)
            embeddedSpeaker = self.speakerEmb(l_)
            cFeature = torch.cat([cFeature, embeddedSpeaker], dim=2)

        predictions = self.wPrediction(cFeature, sampledData)

        outLosses = [0 for x in range(self.nPredicts)]
        outAcc = [0 for x in range(self.nPredicts)]

        for k, locPreds in enumerate(predictions[:self.nPredicts]):
            locPreds = locPreds.permute(0, 2, 1)
            locPreds = locPreds.contiguous().view(-1, locPreds.size(2))
            lossK = self.lossCriterion(locPreds, labelLoss)
            outLosses[k] += lossK.view(1, -1)
            _, predsIndex = locPreds.max(1)
            outAcc[k] += torch.sum(predsIndex == labelLoss).float().view(1, -1)

        return torch.cat(outLosses, dim=1), \
            torch.cat(outAcc, dim=1) / (windowSize * batchSize)


class SpeakerCriterion(BaseCriterion):

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

        loss = self.lossCriterion(predictions, label).view(1, -1)
        acc = (predictions.max(1)[1] == label).double().mean().view(1, -1)

        return loss, acc


class PhoneCriterion(BaseCriterion):

    def __init__(self, dimEncoder, nPhones, onEncoder,
                 nLayers=1):

        super(PhoneCriterion, self).__init__()
        if nLayers == 1:
            self.PhoneCriterionClassifier = nn.Linear(dimEncoder, nPhones)
        else:
            outLayers = [nn.Linear(dimEncoder, nPhones)]
            for l in range(nLayers - 1):
                outLayers.append(nn.ReLU())
                outLayers.append(nn.Linear(nPhones, nPhones))
            self.PhoneCriterionClassifier = nn.Sequential(*outLayers)

        self.lossCriterion = nn.CrossEntropyLoss()
        self.onEncoder = onEncoder

    def forward(self, cFeature, otherEncoded, label):

        # cFeature.size() : batchSize x seq Size x hidden size
        if self.onEncoder:
            predictions = self.getPrediction(otherEncoded)
        else:
            predictions = self.getPrediction(cFeature)
        predictions = predictions.view(-1, predictions.size(2))
        label = label.view(-1)
        loss = self.lossCriterion(predictions, label).view(1, -1)
        acc = (predictions.max(1)[1] == label).double().mean().view(1, -1)
        return loss, acc

    def getPrediction(self, cFeature):
        batchSize, seqSize = cFeature.size(0), cFeature.size(1)
        cFeature = cFeature.contiguous().view(batchSize * seqSize, -1)
        output = self.PhoneCriterionClassifier(cFeature)
        return output.view(batchSize, seqSize, -1)


class CTCPhoneCriterion(BaseCriterion):

    def __init__(self, dimEncoder, nPhones, onEncoder):

        super(CTCPhoneCriterion, self).__init__()
        self.PhoneCriterionClassifier = nn.Linear(dimEncoder, nPhones + 1)
        self.lossCriterion = nn.CTCLoss(blank=nPhones, zero_infinity=True)
        self.onEncoder = onEncoder
        if onEncoder:
            raise ValueError("On encoder version not implemented yet")
        self.BLANK_LABEL = nPhones

    def getPrediction(self, cFeature):
        B, S, H = cFeature.size()
        cFeature = cFeature.contiguous().view(B*S, H)
        return self.PhoneCriterionClassifier(cFeature).view(B, S, -1)

    def forward(self, cFeature, otherEncoded, label):

        # cFeature.size() : batchSize x seq Size x hidden size
        B, S, H = cFeature.size()
        predictions = self.getPrediction(cFeature)
        label = label.to(predictions.device)
        label,  sizeLabels = collapseLabelChain(label)

        avgPER = 0.
        predictions = torch.nn.functional.log_softmax(predictions, dim=2)
        predictions = predictions.permute(1, 0, 2)
        targetSizePred = torch.ones(B, dtype=torch.int64,
                                    device=predictions.device) * S
        loss = self.lossCriterion(predictions, label,
                                  targetSizePred, sizeLabels).view(1, -1)

        return loss, avgPER * torch.ones(1, 1, device=loss.device)


class ModelCriterionCombined(torch.nn.Module):
    def __init__(self, model, criterion):
        super(ModelCriterionCombined, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, data, label):
        c_feature, encoded_data, label = self.model(data, label)
        loss, acc = self.criterion(c_feature, encoded_data, label)
        return loss, acc
