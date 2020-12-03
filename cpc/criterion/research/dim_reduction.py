# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
import argparse
import torch
import progressbar
from random import shuffle


class PCA(torch.nn.Module):

    def __init__(self, k):
        super(PCA, self).__init__()
        self.building = True
        self.register_buffer('var', torch.zeros(k, k))
        self.register_buffer('mean', torch.zeros(k))
        self.register_buffer('PCA_mul', torch.zeros(1, k, k))
        self.register_buffer('PCA_values', torch.zeros(k))
        self.N = 0

    def build(self, normalize=True):

        self.normalize = normalize
        self.var /= self.N
        self.mean /= self.N
        self.var = self.var - self.mean.view(1, -1) * self.mean.view(-1, 1)
        k = self.var.size(0)

        e_vals, e_vects = torch.symeig(self.var, eigenvectors=True)
        self.PCA_mul = e_vects.t().view(1, k, k).clone()
        self.PCA_values = e_vals.clone()
        self.building = False

    def update(self, x):

        if len(x.size()) == 3:
            x = x.contiguous().view(-1, x.size(2))
        assert(len(x.size()) == 2)
        assert(x.size(1) == self.mean.size(0))
        N, k = x.size()
        with torch.no_grad():
            self.var += (x.view(N, 1, k) * x.view(N, k, 1)).sum(dim=0)
            self.mean += x.sum(dim=0)
            self.N += N

    def forward(self, x):

        reshape = False
        if len(x.size()) == 3:
            b, s, _ = x.size()
            x = x.contiguous().view(b*s, -1)
            reshape = True
        assert(not self.building)
        N, k = x.size()
        x -= self.mean
        x = (self.PCA_mul * x.view(N, 1, k)).sum(dim=2)
        if reshape:
            x = x.view(b, s, -1)
        return x


class SFALinear(torch.nn.Module):

    def __init__(self, k):
        super(SFALinear, self).__init__()
        self.register_buffer('covar_speed', torch.zeros(k, k))
        self.register_buffer('mean_x', torch.zeros(k))
        self.register_buffer('square_x', torch.zeros(k))
        self.register_buffer('covar_x', torch.zeros(k, k))
        self.register_buffer('normalizer', torch.zeros(1, k, k))
        self.register_buffer('PCA_mul', torch.zeros(1, k, k))
        self.register_buffer('PCA_values', torch.zeros(k))
        self.register_buffer('projection', torch.zeros(1, k, k))
        self.N_speed = 0
        self.N_x = 0
        self.k = k
        self.building = True

    def update(self, x):

        assert(len(x.size()) == 3)
        assert(x.size(2) == self.mean_x.size(0))
        N, S, k = x.size()
        x = x[:, 1:]
        with torch.no_grad():
            xt = (x[:, 1:] - x[:, :-1]).contiguous().view(-1, k)
            self.covar_speed += (xt.view(-1, 1, k) *
                                 xt.view(-1, k, 1)).sum(dim=0)
            self.N_speed += N * (S-1)

            self.mean_x += x.sum(dim=0).sum(dim=0)
            self.square_x += (x**2).sum(dim=0).sum(dim=0)
            xp = x.contiguous()
            self.covar_x += (xp.view(-1, 1, k) * xp.view(-1, k, 1)).sum(dim=0)
            self.N_x += N * S

    def build(self):

        self.mean_x /= self.N_x
        self.covar_x /= self.N_x
        self.covar_x = self.covar_x - \
            self.mean_x.view(1, -1) * self.mean_x.view(-1, 1)

        # Variance around each dimension
        self.square_x /= self.N_x
        self.square_x = \
            torch.sqrt(torch.clamp(self.square_x - self.mean_x * self.mean_x,
                                   min=0))
        inv_square_x = 1 / (self.square_x + 1e-08)

        # Covariance matrix of the noramlized value of x
        covar_x_normalized = inv_square_x.view(-1, 1) * \
            self.covar_x * inv_square_x.view(1, -1)

        # Inverse Cholesky decomposition of the normalized matrix
        # l l.t = covar_x_normalized
        l_ = torch.cholesky(covar_x_normalized).inverse().detach()
        self.normalizer = l_.view(1, self.k, self.k).clone()

        # Now build the speed covariance matrix
        self.covar_speed /= self.N_speed
        self.covar_speed = inv_square_x.view(-1, 1) * \
            self.covar_speed * inv_square_x.view(1, -1)
        self.covar_speed = torch.mm(l_, torch.mm(self.covar_speed, l_.t()))

        e_vals, e_vects = torch.symeig(self.covar_speed, eigenvectors=True)
        k = e_vects.size(0)
        self.PCA_mul = e_vects.t().view(1, k, k).clone()
        self.PCA_values = e_vals.clone()
        self.building = False
        self.projection = self.PCA_mul.clone()

    def selectDimensions(self, indexVector):
        self.projection = self.PCA_mul[0, indexVector > 0].view(1, -1, self.k)

    def forward(self, x):

        assert(not self.building)
        N, S, k = x.size()
        x = x.contiguous().view(-1, k)
        x -= self.mean_x.view(1, -1)
        x /= (self.square_x.view(1, -1) + 1e-08)
        x = (self.normalizer * x.view(N * S, 1, k)).sum(dim=2)
        x = (self.projection * x.view(N * S, 1, k)).sum(dim=2)
        return x.view(N, S, -1)


def buildPCA(dataLoader, featureMaker, k, normalize=False):

    output_PCA = PCA(k)
    output_PCA.cuda()

    print("Performing the PCA...")
    bar = progressbar.ProgressBar(len(dataLoader))
    bar.start()
    for index, data in enumerate(dataLoader):
        bar.update(index)
        output_PCA.update(featureMaker(data))

    bar.finish()
    output_PCA.build(normalize=normalize)
    return output_PCA


def buildSFA(dataLoader, featureMaker, k):

    output_SFA = SFALinear(k)
    output_SFA.cuda()
    featureMaker.collapse = False

    print("Performing the SFA...")
    bar = progressbar.ProgressBar(len(dataLoader))
    bar.start()
    for index, data in enumerate(dataLoader):
        bar.update(index)
        output_SFA.update(featureMaker(data))

    bar.finish()
    output_SFA.build()
    return output_SFA


def loadDimReduction(path, centroidLimits):
    state_dict = torch.load(path)
    if state_dict["type"] == "PCA":
        dimRed = PCA(state_dict["inDim"])
    elif state_dict["type"] == "SFA":
        dimRed = SFALinear(state_dict["inDim"])
    else:
        raise ValueError(f"Invalid module type {state_dict['type']}")
    dimRed.load_state_dict(state_dict["state_dict"])
    dimRed.building = False
    if centroidLimits is not None:
        centroidsVals = state_dict["centroid_values"]
        dimRed.selectDimensions(
            (centroidsVals > centroidLimits[0]) *
            (centroidsVals < centroidLimits[1]))
    return dimRed


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from dataset import findAllSeqs, filterSeqs, AudioBatchData
    from train import loadModel, getCheckpointData
    from feature_maker import FeatureModule

    parser = argparse.ArgumentParser(description='Dim reduction. Performing \
                                                  geither a PCA or a SFA')
    parser.add_argument('pathCheckpoint', type=str)
    parser.add_argument('pathOut', type=str)
    parser.add_argument(
        '--pathDB', type=str,
        default="/datasets01_101/LibriSpeech/022219/train-clean-100/")
    parser.add_argument('--seqList', type=str,
                        default='/private/home/mriviere/LibriSpeech/LibriSpeech100_labels_split/train_split.txt')
    parser.add_argument('--recursionLevel', type=int, default=2)
    parser.add_argument('--extension', type=str, default='.flac')
    parser.add_argument('--mode', type=str, default='SFA',
                        choices=['PCA', 'SFA'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--batchSize', type=int, default=8)
    parser.add_argument('--sizeWindow', type=int, default=20480)
    args = parser.parse_args()

    seqNames, speakers = findAllSeqs(args.pathDB,
                                     recursionLevel=args.recursionLevel,
                                     extension=args.extension)

    if args.seqList is not None:
        seqNames = filterSeqs(args.seqList, seqNames)
    if args.debug:
        shuffle(seqNames)
        seqNames = seqNames[:100]

    dataset = AudioBatchData(args.pathDB,
                             args.sizeWindow,
                             seqNames,
                             None,
                             list(speakers))
    trainLoader = dataset.getDataLoader(args.batchSize, "sequential", False)

    # Load the model
    featureMaker = loadModel([args.pathCheckpoint])[0]
    featureMaker.gAR.keepHidden = True
    featureMaker = FeatureModule(featureMaker, False).cuda()

    # Get the output dimension
    modelArgs = getCheckpointData(os.path.dirname(args.pathCheckpoint))[2]
    outDim = modelArgs.hiddenGar

    if args.mode == 'SFA':
        featureMaker.collapse = False
        dim_reduction = buildSFA(
            trainLoader, featureMaker, outDim)
    else:
        dim_reduction = buildPCA(trainLoader, featureMaker, outDim)

    out_state_dict = {"state_dict": dim_reduction.state_dict(),
                      "inDim": outDim,
                      "type": args.mode}
    torch.save(out_state_dict, args.pathOut)
    pathArgs = f"{os.path.splitext(args.pathOut)[0]}_args.json"
    with open(pathArgs, 'w') as file:
        json.dump(vars(args), file, indent=2)
