import torch
import torch.nn as nn
import numpy as np
from clustering import kMeanCluster, kMeanGPU, FeatureModule, \
    distanceEstimation, fastDPMean
from beam_search import collapseLabelChain


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
        for k in range(len(self.predictors)):

            locC = self.predictors[k](c)
            if isinstance(locC, tuple):
                locC = locC[0]
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
                                     * windowSize * batchSize, ), device=encodedData.device)

        baseIdx = torch.arange(0, windowSize, device=encodedData.device)
        baseIdx = baseIdx.view(1, 1, windowSize).expand(1,
                                                        self.negativeSamplingExt, windowSize).expand(batchSize, self.negativeSamplingExt, windowSize)
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


class CPCBertCriterion(BaseCriterion):

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


class AdvSpeakerCriterion(BaseCriterion):

    def __init__(self, dimEncoder, nSpeakers, onEncoder):

        super(AdvSpeakerCriterion, self).__init__()
        self.linearSpeakerClassifier = nn.Linear(
            dimEncoder, nSpeakers)
        self.lossCriterion = nn.CrossEntropyLoss()
        self.entropyCriterion = nn.LogSoftmax(dim=1)
        self.onEncoder = onEncoder
        self.softMax = nn.Softmax(dim=1)
        print(f"{nSpeakers} found")

    def forward(self, cFeature, otherEncoded, label):

        # cFeature.size() : batchSize x seq Size x hidden size
        if self.onEncoder:
            features = otherEncoded
        else:
            features = cFeature

        B, S, H = features.size()
        features = features.mean(dim=1)
        predictions = self.linearSpeakerClassifier(features)
        if label is None:
            loss = (self.entropyCriterion(predictions) * self.softMax(predictions)).sum(dim=1).view(-1)
            acc = torch.zeros(1, 1).cuda()
        else:
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


###################
# CLUSTERING LOSS
###################

class ClusteringLoss(torch.nn.Module):

    def __init__(self, k, d, delay, clusterIter, clusteringUpdate):

        super(ClusteringLoss, self).__init__()
        self.clusters = kMeanCluster(torch.zeros(1, k, d))
        self.k = k
        self.d = d
        self.init = False
        self.delay = delay
        self.step = 0
        self.clusterIter = clusterIter

        self.TARGET_QUANTILE = 0.05
        availableUpdates = ['kmean', 'dpmean']
        if clusteringUpdate not in availableUpdates:
            raise ValueError(f"{clusteringUpdate} is an invalid clustering \
                            update option. Must be in {availableUpdates}")

        print(f"Clustering update mode is {clusteringUpdate}")
        self.DP_MEAN = clusteringUpdate == 'dpmean'

    def canRun(self):

        return self.step > self.delay

    def getOPtimalLambda(self, dataLoader, model, MAX_ITER=10):

        distData = distanceEstimation(model, dataLoader, maxIndex=MAX_ITER,
                                      maxSizeGroup=300)
        nData = len(distData)
        print(f"{nData} samples analyzed")
        index = int(self.TARGET_QUANTILE * nData)
        return distData[index]

    def updateCLusters(self, dataLoader, model, MAX_ITER=20, EPSILON=1e-4):

        self.step += 1
        if not self.canRun():
            return

        featureMaker = FeatureModule(model, False).cuda()
        featureMaker = torch.nn.DataParallel(featureMaker)
        if self.DP_MEAN:
            l_ = self.getOPtimalLambda(dataLoader, featureMaker)
            clusters = fastDPMean(dataLoader, featureMaker,
                                  l_,
                                  MAX_ITER=MAX_ITER,
                                  perIterSize=self.clusterIter)
            self.k = clusters.size(1)
        else:
            start_clusters = None
            clusters = kMeanGPU(dataLoader, featureMaker, self.k,
                                MAX_ITER=MAX_ITER, EPSILON=EPSILON,
                                perIterSize=self.clusterIter,
                                start_clusters=start_clusters)
        self.clusters = kMeanCluster(clusters)
        self.init = True


class DeepClustering(ClusteringLoss):

    def __init__(self, *args):
        ClusteringLoss.__init__(self, *args)
        self.classifier = nn.Linear(self.d, self.k)
        self.lossCriterion = nn.CrossEntropyLoss()

    def forward(self, x, labels):

        if not self.canRun():
            return torch.zeros(1, 1, device=x.device)

        B, S, D = x.size()
        predictedLabels = self.classifier(x.view(-1, D))

        return self.lossCriterion(predictedLabels,
                                  labels.view(-1)).mean().view(-1, 1)


class CTCCLustering(ClusteringLoss):
    def __init__(self, *args):
        ClusteringLoss.__init__(self, *args)
        self.mainModule = CTCPhoneCriterion(self.d, self.k, False)

    def forward(self, cFeature, label):
        return self.mainModule(cFeature, None, label)[0]


class DeepEmbeddedClustering(ClusteringLoss):

    def __init__(self, lr, *args):

        self.lr = lr
        ClusteringLoss.__init__(self, *args)

    def forward(self, x):

        if not self.canRun():
            return torch.zeros(1, 1, device=x.device)

        B, S, D = x.size()
        clustersDist = self.clusters(x)
        clustersDist = clustersDist.view(B*S, -1)
        clustersDist = 1.0 / (1.0 + clustersDist)
        Qij = clustersDist / clustersDist.sum(dim=1, keepdim=True)

        qFactor = (Qij**2) / Qij.sum(dim=0, keepdim=True)
        Pij = qFactor / qFactor.sum(dim=1, keepdim=True)

        return (Pij * torch.log(Pij / Qij)).sum().view(1, 1)

    def updateCLusters(self, dataLoader, model):

        if not self.init:
            super(DeepEmbeddedClustering, self).updateCLusters(
                dataLoader, model)
            self.clusters.Ck.requires_grad = True
            self.init = True
            return

        self.step += 1
        if not self.canRun():
            return

        print("Updating the deep embedded clusters")
        optimizer = torch.optim.SGD([self.clusters.Ck], lr=self.lr)

        maxData = len(
            dataLoader) if self.clusterIter <= 0 else self.clusterIter

        for index, data in enumerate(dataLoader):
            if index > maxData:
                break

            optimizer.zero_grad()

            batchData, label = data
            batchData = batchData.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            with torch.no_grad():
                cFeature, _, _ = model(batchData, label)

            loss = self.forward(cFeature).sum()
            loss.backward()

            optimizer.step()
