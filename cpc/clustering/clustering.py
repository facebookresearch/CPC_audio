# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import progressbar
import torch
import torch.nn as nn
import cpc.feature_loader as fl
from os.path import join
from os import remove
from time import time
from pathlib import Path


def loadClusterModule(pathCheckpoint):
    print(f"Loading ClusterModule at {pathCheckpoint}")
    state_dict = torch.load(pathCheckpoint)
    clusterModule = kMeanCluster(state_dict["state_dict"]["Ck"])
    clusterModule = clusterModule.cuda()
    return clusterModule


class kMeanCluster(nn.Module):
    def __init__(self, Ck):

        super(kMeanCluster, self).__init__()
        self.register_buffer("Ck", Ck)
        self.k = Ck.size(1)

    def forward(self, features):
        B, S, D = features.size()
        features = features.contiguous().view(B * S, 1, -1)
        return ((features - self.Ck) ** 2).sum(dim=2).view(-1, S, self.k)


class kMeanClusterStep(torch.nn.Module):
    def __init__(self, k, D):

        super(kMeanClusterStep, self).__init__()
        self.k = k
        self.register_buffer("Ck", torch.zeros(1, k, D))

    def forward(self, locF):

        index = ((locF - self.Ck) ** 2).mean(dim=2).min(dim=1)[1]
        Ck1 = torch.cat(
            [locF[index == p].sum(dim=0, keepdim=True) for p in range(self.k)], dim=1
        )
        nItems = torch.cat(
            [(index == p).sum(dim=0, keepdim=True) for p in range(self.k)], dim=0
        ).view(1, -1)
        return Ck1, nItems


class DPMeanClusterStep(torch.nn.Module):
    def __init__(self, mu):

        super(DPMeanClusterStep, self).__init__()
        self.k = 1
        self.register_parameter("Ck", torch.nn.Parameter(mu))

    def forward(self, features):

        distance, index = (features - self.mu).norm(dim=2).min(dim=1)
        maxDist = distance.max().view(1)
        return distance, index, maxDist


def buildNewPhoneDict(pathDIR, seqNames, model, clusters, nk):

    featureMaker = fl.FeatureModule(model, False)
    featureMaker = fl.ModelClusterCombined(featureMaker, clusters, nk, "int")
    featureMaker.cuda()

    outDict = {}
    fillingStatus = torch.zeros(nk, dtype=torch.long)

    print("Building the new features labels from clusters...")
    for seqPath in seqNames:
        fullPath = os.path.join(pathDIR, seqPath)
        with torch.no_grad():
            features = fl.buildFeature(featureMaker, fullPath, strict=True)
            oneHotFeatures = fl.toOneHot(features, nk).view(-1, nk)
            fillingStatus += oneHotFeatures.sum(dim=0)
        outDict[os.path.splitext(os.path.basename(seqPath))[0]] = features.view(
            -1
        ).tolist()
    print("...done")
    return outDict, fillingStatus


def save_cluster_step(
    Ck: torch.tensor,
    path_out: Path,
    mode: str = None,
    iter: int = None,
    last_diff: float = None,
):
    out_state_dict = {}
    out_state_dict["state_dict"] = {"Ck": Ck}
    out_state_dict["n_clusters"] = Ck.size(1)
    out_state_dict["dim"] = Ck.size(2)
    out_state_dict["iteration"] = iter
    out_state_dict["last_diff"] = last_diff
    out_state_dict["mode"] = mode
    torch.save(out_state_dict, path_out)


def get_last_checkpoint(path_in: Path):

    checkpoint_list = list(Path(path_in).glob("checkpoint_*.pt"))
    valid_paths = [x for x in checkpoint_list if x.stem.split("_")[-1].isdigit()]
    valid_paths.sort(key=lambda x: int(x.stem.split("_")[-1]))
    if len(valid_paths) == 0:
        raise RuntimeError("No checkpoint found")
    return valid_paths[-1]


def setup_log_file(save_dir: Path):
    path_logs = Path(save_dir) / "training_logs.txt"
    return logging.FileHandler(path_logs)


def kMeanGPU(
    dataLoader,
    featureMaker,
    k,
    n_group=1,
    MAX_ITER=100,
    EPSILON=1e-4,
    perIterSize=-1,
    start_clusters=None,
    save_dir=None,
    save_last=5,
):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Kmean")
    save = save_dir is not None
    if save:
        save_dir = Path(save_dir)
        logger.addHandler(setup_log_file(save_dir))

    logger.info(f"Start Kmean clustering with {k} clusters and {n_group} groups...")

    if start_clusters is None:
        Ck = []
        with torch.no_grad():
            for index, data in enumerate(dataLoader):
                cFeature = featureMaker(data)
                cFeature = cFeature.contiguous().view(-1, cFeature.size(2) // n_group)
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

    if perIterSize < 0:
        perIterSize = len(dataLoader)

    clusterStep = kMeanClusterStep(k, D).cuda()
    clusterStep = torch.nn.DataParallel(clusterStep)
    clusterStep.module.Ck.copy_(Ck)

    bar = progressbar.ProgressBar(maxval=MAX_ITER)
    bar.start()
    iter, stored = 0, 0
    with torch.no_grad():
        while iter < MAX_ITER:
            start_time = time()
            Ck1 = torch.zeros(Ck.size()).cuda()
            nItemsClusters = torch.zeros(Ck.size(1), dtype=torch.long).cuda()
            for index, data in enumerate(dataLoader):
                cFeature = featureMaker(data).contiguous().view(-1, 1, D)
                locC, locN = clusterStep(cFeature)
                Ck1 += locC.sum(dim=0, keepdim=True)
                nItemsClusters += locN.sum(dim=0)

            iter += 1
            bar.update(iter)
            print()

            nItemsClusters = nItemsClusters.float().view(1, -1, 1) + 1e-8
            Ck1 /= nItemsClusters
            last_diff = (clusterStep.module.Ck - Ck1).norm(dim=2).max().item()
            nItems = int(nItemsClusters.sum().item())
            logger.info(
                f"ITER {iter} done in {time()-start_time:.2f} seconds. nItems: {nItems}. Difference with last checkpoint: {last_diff}"
            )

            if save:
                path_save = save_dir / f"checkpoint_{iter}.pt"
                logger.info(f"Saving last checkpoint to {path_save}")
                save_cluster_step(
                    Ck1,
                    path_save,
                    iter=iter,
                    last_diff=last_diff,
                    mode="kMean",
                )
                if (save_dir / f"checkpoint_{iter-save_last}.pt").is_file():
                    remove(save_dir / f"checkpoint_{iter-save_last}.pt")
            if last_diff < EPSILON:
                logger.info(f"Clustering ended in {iter} iterations out of {MAX_ITER}")
                break
            clusterStep.module.Ck.copy_(Ck1)

    bar.finish()

    logger.info(f"Clustering ended in {MAX_ITER} iterations out of {MAX_ITER}")
    logger.info(f"Last diff {last_diff}")
    if start_clusters is not None:
        nEmptyClusters = (nItemsClusters < 1).sum().item()
        logger.info(f"{nEmptyClusters} empty clusters out of {k}")
    return clusterStep.module.Ck


def fastDPMean(
    dataLoader,
    featureMaker,
    l,
    MAX_ITER=100,
    batchSize=1000,
    EPSILON=1e-4,
    perIterSize=-1,
    save_dir=None,
    save_last=5,
    mu_start=None,
):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DPMean")
    save = save_dir is not None
    if save:
        save_dir = Path(save_dir)
        logger.addHandler(setup_log_file(save_dir))

    logger.info(f"{perIterSize} updates per iteration")

    bar = progressbar.ProgressBar(maxval=MAX_ITER)
    bar.start()
    with torch.no_grad():
        if mu_start is not None:
            mu = mu_start.clone()
            _, k, D = mu.size()
        else:
            print("Start training from scratch. Creating new mu ...")
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
            mu /= nSeqs

        def resetTmpData():
            mu1 = torch.zeros(mu.size()).cuda()
            c1 = torch.zeros(mu.size(1), dtype=torch.long).cuda()
            return mu1, c1

        mu1, c1 = resetTmpData()
        storedData = 0

        iter = 0
        while iter < MAX_ITER:
            start_time = time()
            for nBatch, data in enumerate(dataLoader):
                features = featureMaker(data)
                N, S, _ = features.size()
                features = features.contiguous().view(N * S, D).view(-1, 1, D)
                distance, index = (features - mu).norm(dim=2).min(dim=1)
                maxDist = distance.max()

                if maxDist > l:
                    indexFeature = distance.argmax()
                    mu = torch.cat([mu, features[indexFeature].view(1, 1, D)], dim=1)
                    mu1 = torch.cat(
                        [mu1, torch.zeros(1, 1, D, device=mu.device)], dim=1
                    )
                    c1 = torch.cat(
                        [c1, torch.zeros(1, device=mu.device, dtype=torch.long)], dim=0
                    )
                    index[indexFeature] = k
                    k += 1
                    if k % 10 == 0:
                        logger.info(f"Number of clusters increased to {k}")

                mu1 += torch.cat(
                    [features[index == p].sum(dim=0, keepdim=True) for p in range(k)],
                    dim=1,
                )
                c1 += torch.cat(
                    [(index == p).sum(dim=0, keepdim=True) for p in range(k)], dim=0
                )

            c1 = c1.float().view(1, -1, 1) + 1e-4
            mu1 /= c1
            last_diff = (mu - mu1).norm(dim=2).max().item()
            nItems = int(c1.sum().cpu().detach().item())

            mu = mu1
            k = mu.size(1)
            mu1, c1 = resetTmpData()

            iter += 1
            bar.update(iter)
            print()

            logger.info(
                f"ITER {iter} done in {time()-start_time:.2f} seconds. nItems: {nItems}. lambda={l}. mu shape: {mu.size()}. Difference with last checkpoint: {last_diff}"
            )

            if save:
                path_save = save_dir / f"checkpoint_{iter}.pt"
                logger.info(f"Saving last checkpoint to {path_save}")
                save_cluster_step(
                    mu,
                    path_save,
                    iter=iter,
                    last_diff=last_diff,
                    mode="DPMean",
                )
                if (save_dir / f"checkpoint_{iter-save_last}.pt").is_file():
                    remove(save_dir / f"checkpoint_{iter-save_last}.pt")

            if last_diff < EPSILON:
                logger.info(f"Clustering ended in {iter} iterations out of {MAX_ITER}")
                break

    bar.finish()

    _, k, D = mu.size()
    logger.info(f"{k} clusters found for lambda = {l}")
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
            Ck1 = torch.cat(
                [C[I == p].mean(dim=0, keepdim=True) for p in range(k)], dim=1
            )
            last_diff = (Ck - Ck1).norm(dim=2).max().item()
            if last_diff < EPSILON:
                print(f"Clustering ended in {iter} iterations out of {MAX_ITER}")
                break
            Ck = Ck1

    bar.finish()
    print(f"Clustering ended in {MAX_ITER} iterations out of {MAX_ITER}")
    print(f"Last diff {last_diff}")
    return Ck


def distanceEstimation(featureMaker, dataLoader, maxIndex=10, maxSizeGroup=300):

    outData = []
    maxIndex = min(maxIndex, len(dataLoader))

    print("Computing the features...")
    for index, item in enumerate(dataLoader):

        with torch.no_grad():
            features = featureMaker(item).cpu()

        N, S, C = features.size()
        outData.append(features.contiguous().view(N * S, C))

        if index > maxIndex:
            break
    print("Done")
    outData = torch.cat(outData, dim=0)
    nItems = outData.size(0)
    outData = outData[torch.randperm(nItems)]

    maxIter = nItems // maxSizeGroup
    if maxIter * maxSizeGroup < nItems:
        maxIter += 1

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
