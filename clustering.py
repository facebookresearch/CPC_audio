import torch
import numpy as np
import time
import argparse
import sys
import progressbar
import os
import json
from random import shuffle
from dataset import findAllSeqs, filterSeqs, AudioBatchData
from dim_reduction import loadDimReduction
from feature_maker import FeatureModule
from plot import plotHist


class kMeanCluster(torch.nn.Module):

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
                    mu1 = torch.cat([mu1, torch.zeros(1, 1, D, device=mu.device)], dim=1)
                    c1 = torch.cat([c1, torch.zeros(1, device=mu.device, dtype=torch.long)], dim=0)
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


def getQuantile(sortedData, percent):
    return sortedData[int(percent * len(sortedData))]


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



def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('pathCheckpoint', type=str)
    parser.add_argument('pathOutput', type=str)
    parser.add_argument(
        '--pathDB', type=str,
        default="/datasets01_101/LibriSpeech/022219/train-clean-100/")
    parser.add_argument('-k', '--nClusters', type=int, default=50)
    parser.add_argument('-n', '--MAX_ITER', type=int, default=100)
    parser.add_argument('--recursionLevel', type=int, default=2)
    parser.add_argument('--extension', type=str, default='.flac')
    parser.add_argument('--seqList', type=str, default=None)
    parser.add_argument('--sizeWindow', type=int, default=10240)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--encoder_layer', action='store_true')
    parser.add_argument('--batchSizeGPU', type=int, default=50)
    parser.add_argument('--DPMean', action='store_true')
    parser.add_argument('--DPLambda', type=float, default=11)
    parser.add_argument('--perIterSize', type=int, default=-1)
    parser.add_argument('--train_mode', action='store_true')
    parser.add_argument('--dimReduction', type=str, default=None)
    parser.add_argument('--centroidLimits', type=int, nargs=2, default=None)
    parser.add_argument('--getDistanceEstimation', action='store_true')
    return parser.parse_args(argv)


if __name__ == "__main__":
    from train import loadModel

    args = parseArgs(sys.argv[1:])
    seqNames, speakers = findAllSeqs(args.pathDB,
                                     recursionLevel=args.recursionLevel,
                                     extension=args.extension)

    if args.seqList is not None:
        seqNames = filterSeqs(args.seqList, seqNames)
    if args.debug:
        shuffle(seqNames)
        seqNames = seqNames[:100]
    if args.getDistanceEstimation:
        shuffle(seqNames)
        seqNames = seqNames[:5000]

    dataset = AudioBatchData(args.pathDB,
                             args.sizeWindow,
                             seqNames,
                             None,
                             list(speakers))

    nGPUs = torch.cuda.device_count()
    batchSize = args.batchSizeGPU * nGPUs
    trainLoader = dataset.getDataLoader(batchSize, "uniform",
                                        False, numWorkers=0)

    featureMaker = loadModel([args.pathCheckpoint])[0]
    featureMaker = FeatureModule(featureMaker, args.encoder_layer)

    if args.dimReduction is not None:
        dimRed = loadDimReduction(args.dimReduction, args.centroidLimits)
        featureMaker = torch.nn.Sequential(featureMaker, dimRed)
    if not args.train_mode:
        featureMaker.eval()
    featureMaker.cuda()

    pathConfig = f"{os.path.splitext(args.pathOutput)[0]}_args.json"
    with open(pathConfig, 'w') as file:
        json.dump(vars(args), file, indent=2)

    if args.getDistanceEstimation:
        print("Performing the estimetion of the distance distribution \
               between features")
        distRepartition = distanceEstimation(featureMaker, trainLoader)
        args.pathOutput = os.path.splitext(args.pathOutput)[0]

        if not os.path.isdir(args.pathOutput):
            os.mkdir(args.pathOutput)

        outDict = {x: getQuantile(distRepartition, x)
                   for x in np.arange(0, 1., 0.1)}

        pathDict = os.path.join(args.pathOutput, "quantiles.json")
        with open(pathDict, 'w') as f:
            json.dump(outDict, f, indent=2)

        distRepartition = np.array(distRepartition)

        pathHist = os.path.join(args.pathOutput, "distance_distribution.png")
        plotHist(distRepartition, 300, pathHist)

        pathRaw = os.path.join(args.pathOutput, "raw.npy")
        with open(pathRaw, 'wb') as f:
            np.save(f, distRepartition)

        sys.exit()


    out_state_dict = {}
    print("Starting the clustering...")
    start_time = time.time()
    if args.DPMean:
        clusters = fastDPMean(trainLoader, featureMaker,
                              args.DPLambda,
                              MAX_ITER=args.MAX_ITER,
                              perIterSize=args.perIterSize).cpu()
        args.nClusters = clusters.size(1)
    else:
        clusters = kMeanGPU(trainLoader, featureMaker, args.nClusters,
                            perIterSize=args.perIterSize,
                            MAX_ITER=args.MAX_ITER).cpu()

    print(f'Ran clustering '
          f'in {time.time() - start_time:.2f} seconds')

    clusterModule = kMeanCluster(clusters)
    out_state_dict["state_dict"] = clusterModule.state_dict()
    out_state_dict["encoder_layer"] = args.encoder_layer
    out_state_dict["n_clusters"] = args.nClusters
    out_state_dict['dim'] = clusters.size(2)
    torch.save(out_state_dict, args.pathOutput)
    with open(pathConfig, 'w') as file:
        json.dump(vars(args), file, indent=2)
