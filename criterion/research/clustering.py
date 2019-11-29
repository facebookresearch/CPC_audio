import progressbar
import torch
import torch.nn as nn
from .. import CTCPhoneCriterion


class kMeanCluster(nn.Module):

    def __init__(self, Ck):

        super(kMeanCluster, self).__init__()
        self.register_buffer('Ck', Ck)
        self.k = Ck.size(1)

    def forward(self, features):
        B, S, D = features.size()
        features = features.contiguous().view(B*S, 1, -1)
        return ((features - self.Ck)**2).sum(dim=2).view(-1, S, self.k)


class kMeanClusterStep(torch.nn.Module):

    def __init__(self, k, D):

        super(kMeanClusterStep, self).__init__()
        self.k = k
        self.register_buffer('Ck', torch.zeros(1, k, D))

    def forward(self, locF):

        index = ((locF - self.Ck)**2).mean(dim=2).min(dim=1)[1]
        Ck1 = torch.cat([locF[index == p].sum(dim=0, keepdim=True)
                         for p in range(self.k)], dim=1)
        nItems = torch.cat([(index == p).sum(dim=0, keepdim=True)
                            for p in range(self.k)], dim=0).view(1, -1)
        return Ck1, nItems


class DPMeanClusterStep(torch.nn.Module):

    def __init__(self, mu):

        super(DPMeanClusterStep, self).__init__()
        self.k = 1
        self.register_parameter('Ck', torch.nn.Parameter(mu))

    def forward(self, features):

        distance, index = (features - self.mu).norm(dim=2).min(dim=1)
        maxDist = distance.max().view(1)
        return distance, index, maxDist


def kMeanGPU(dataLoader, featureMaker, k,
             MAX_ITER=100, EPSILON=1e-4,
             perIterSize=-1, start_clusters = None):

    if start_clusters is None:
        Ck = []
        if perIterSize < 0:
            perIterSize = len(dataLoader)
        with torch.no_grad():
            for index, data in enumerate(dataLoader):
                cFeature = featureMaker(data)
                cFeature = cFeature.contiguous().view(-1, cFeature.size(2))
                Ck.append(cFeature)
                if index > k:
                    break
        Ck = torch.cat(Ck, dim=0)
        N, D = Ck.size()
        indexes = torch.randperm(N)[:k]
        Ck = Ck[indexes].view(1, k, D)
    else:
        Ck = start_clusters
        D = Ck.size(2)

    clusterStep = kMeanClusterStep(k, D).cuda()
    clusterStep = torch.nn.DataParallel(clusterStep)
    clusterStep.module.Ck.copy_(Ck)

    bar = progressbar.ProgressBar(maxval=MAX_ITER)
    bar.start()
    iter, stored = 0, 0
    with torch.no_grad():
        while iter < MAX_ITER:
            Ck1 = torch.zeros(Ck.size()).cuda()
            nItemsClusters = torch.zeros(Ck.size(1),
                                         dtype=torch.long).cuda()
            for index, data in enumerate(dataLoader):
                cFeature = featureMaker(data).contiguous().view(-1, 1, D)
                locC, locN = clusterStep(cFeature)
                Ck1 += locC.sum(dim=0, keepdim=True)
                nItemsClusters += locN.sum(dim=0)
                stored+=1
                if stored >= perIterSize:
                    bar.update(iter)
                    iter+=1
                    stored=0
                    if iter >= MAX_ITER:
                        break

            nItemsClusters = nItemsClusters.float().view(1, -1, 1) + 1e-8
            Ck1 /= nItemsClusters
            lastDiff = (clusterStep.module.Ck - Ck1).norm(dim=2).max().item()
            if lastDiff < EPSILON:
                print(f"Clustering ended in {iter} iterations out of {MAX_ITER}")
                break
            clusterStep.module.Ck.copy_(Ck1)

    bar.finish()

    print(f"Clustering ended in {MAX_ITER} iterations out of {MAX_ITER}")
    print(f"Last diff {lastDiff}")
    if start_clusters is not None:
        nEmptyClusters = (nItemsClusters < 1).sum().item()
        print(f"{nEmptyClusters} empty clusters out of {k}")
    return clusterStep.module.Ck


def fastDPMean(dataLoader, featureMaker, l,
               MAX_ITER=100, batchSize=1000,
               EPSILON=1e-4, perIterSize=-1):

    if perIterSize < 0:
        perIterSize = len(dataLoader)

    print(f"{perIterSize} updates per iteration")

    bar = progressbar.ProgressBar(maxval=MAX_ITER)
    bar.start()
    with torch.no_grad():

        mu = 0
        nSeqs = 100
        for index, data in enumerate(dataLoader):
            features = featureMaker(data)
            mu += features
            if index > nSeqs:
                break

        B, S, D = mu.size()
        k = 1
        mu = mu.contiguous().view(-1, D).mean(dim=0).view(1, k, D)
        mu /= (nSeqs)

        def resetTmpData():
            mu1 = torch.zeros(mu.size()).cuda()
            c1 = torch.zeros(mu.size(1), dtype=torch.long).cuda()
            return mu1, c1

        mu1, c1 = resetTmpData()
        storedData = 0

        iter = 0
        while iter < MAX_ITER:
            for nBatch, data in enumerate(dataLoader):
                features = featureMaker(data)
                N, S, _ = features.size()
                features = features.contiguous().view(N*S, D).view(-1, 1, D)
                distance, index = (features - mu).norm(dim=2).min(dim=1)
                maxDist = distance.max()
                if maxDist > l:
                    indexFeature = distance.argmax()
                    mu = torch.cat([mu, features[indexFeature].view(1, 1, D)],
                                   dim=1)
                    mu1 = torch.cat([mu1, torch.zeros(1, 1, D,
                                                      device=mu.device)],
                                    dim=1)
                    c1 = torch.cat([c1, torch.zeros(1, device=mu.device,
                                                    dtype=torch.long)],
                                    dim=0)
                    index[indexFeature] = k
                    k+=1
                mu1 += torch.cat([features[index == p].sum(dim=0, keepdim=True)
                                 for p in range(k)], dim=1)
                c1 += torch.cat([(index == p).sum(dim=0, keepdim=True)
                                    for p in range(k)], dim=0)
                storedData+=1
                if storedData >= perIterSize:
                    #index = c1 > 0
                    c1 = c1.float().view(1, -1, 1) + 1e-4
                    mu1 /= c1
                    lastDiff = (mu - mu1).norm(dim=2).max().item()
                    #mu1 = mu1[:, index]
                    if lastDiff < EPSILON:
                        print(f"Clustering ended in {iter} iterations out of {MAX_ITER}")
                        break

                    mu = mu1
                    k = mu.size(1)
                    mu1, c1 = resetTmpData()
                    storedData=0
                    bar.update(iter)
                    iter+=1

                    if iter >= MAX_ITER:
                        break

    bar.finish()

    _, k, D = mu.size()
    print(f"{k} clusters found for lambda = {l}")
    return mu


def KMean(C, k, MAX_ITER=100, EPSILON=1e-4, batchSize=1000):

    N, D = C.size()
    indexes = torch.randperm(N)[:k]
    Ck = C[indexes].view(1, k, D)
    C = C.view(N, 1, D)

    bar = progressbar.ProgressBar(maxval=MAX_ITER)
    bar.start()

    with torch.no_grad():
        for iter in range(MAX_ITER):
            bar.update(iter)
            I = (C - Ck).norm(dim=2).min(dim=1)[1]
            Ck1 = torch.cat([C[I == p].mean(dim=0, keepdim=True)
                             for p in range(k)], dim=1)
            lastDiff = (Ck - Ck1).norm(dim=2).max().item()
            if lastDiff < EPSILON:
                print(f"Clustering ended in {iter} iterations out of {MAX_ITER}")
                break
            Ck = Ck1

    bar.finish()
    print(f"Clustering ended in {MAX_ITER} iterations out of {MAX_ITER}")
    print(f"Last diff {lastDiff}")

    return Ck


def distanceEstimation(featureMaker, dataLoader,
                       maxIndex=10, maxSizeGroup=300):

    outData = []
    maxIndex = min(maxIndex, len(dataLoader))

    print("Computing the features...")
    for index, item in enumerate(dataLoader):

        with torch.no_grad():
            features = featureMaker(item).cpu()

        N, S, C = features.size()
        outData.append(features.contiguous().view(N*S, C))

        if index > maxIndex:
            break
    print("Done")
    outData = torch.cat(outData, dim=0)
    nItems = outData.size(0)
    outData = outData[torch.randperm(nItems)]

    maxIter = nItems // maxSizeGroup
    if maxIter * maxSizeGroup < nItems:
        maxIter+=1

    outDist = []
    nVectors = len(outData)

    print("Computing the distance...")
    bar = progressbar.ProgressBar(nVectors)
    bar.start()
    for step in range(maxIter):
        bar.update(step)
        with torch.no_grad():
            minBorn = step * maxSizeGroup
            maxBorn = min(minBorn + maxSizeGroup, nItems)
            sumData = outData[minBorn:maxBorn]
            P, C = sumData.size()
            locDists = (sumData.view(1, P, C) - sumData.view(P, 1, C)).norm(dim=2)
            locDists = locDists[locDists > 0].view(-1)
            outDist += locDists.tolist()

    bar.finish()
    print("Done")
    outDist.sort()
    return outDist


class ClusteringLoss(nn.Module):

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


    def updateCLusters(self, dataLoader, featureMaker,
                       MAX_ITER=20, EPSILON=1e-4):

        self.step += 1
        if not self.canRun():
            return

        featureMaker = featureMaker.cuda()
        if not isinstance(featureMaker, nn.DataParallel):
            featureMaker = nn.DataParallel(featureMaker)

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
