# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
import torch
import progressbar
import argparse
import numpy as np

from cpc.dataset import findAllSeqs
from cpc.criterion.research.clustering import kMeanCluster
from cpc.feature_loader import buildFeature, FeatureModule, \
    ModelPhoneCombined, loadSupervisedCriterion, \
    ModelClusterCombined, loadModel
from cpc.criterion.research.dim_reduction import loadDimReduction


def getArgs(pathCheckpoints):
    pathArgs = os.path.join(os.path.dirname(pathCheckpoints),
                            "checkpoint_args.json")
    with open(pathArgs, 'rb') as file:
        return json.load(file)


def buildAllFeature(featureMaker, pathDB, pathOut,
                    seqList, stepSize=0.01, strict=False,
                    maxSizeSeq=64000, format='fea',
                    seqNorm=False):

    totSeqs = len(seqList)
    startStep = stepSize / 2
    bar = progressbar.ProgressBar(maxval=totSeqs)
    bar.start()
    for nseq, seqPath in enumerate(seqList):
        bar.update(nseq)
        feature = buildFeature(featureMaker,
                               os.path.join(pathDB, seqPath),
                               strict=strict or seqNorm,
                               maxSizeSeq=maxSizeSeq,
                               seqNorm=seqNorm)

        _, nSteps, hiddenSize = feature.size()
        outName = os.path.basename(os.path.splitext(seqPath)[0]) + f'.{format}'
        fname = os.path.join(pathOut, outName)

        if format == 'npz':
            time = [startStep + step * stepSize for step in range(nSteps)]
            values = feature.squeeze(0).float().cpu().numpy()
            totTime = np.array([stepSize * nSteps], dtype=np.float32)
            with open(fname, 'wb') as f:
                np.savez(f, time=time, features=values, totTime=totTime)
        elif format == 'npy':
            time = [startStep + step * stepSize for step in range(nSteps)]
            values = feature.squeeze(0).float().cpu().numpy()
            with open(fname, 'wb') as f:
                np.save(f, values)
        elif format == 'af':
            import arrayfire as af
            time = [startStep + step * stepSize for step in range(nSteps)]
            values = feature.squeeze(0).float().cpu().numpy()
            totTime = np.array([stepSize * nSteps], dtype=np.float32)
            af.save_array("time", af.Array(time, dtype=af.Dtype.f32), fname)
            af.save_array("totTime", af.interop.from_ndarray(totTime),
                          fname, append=True)
            af.save_array("features", af.interop.from_ndarray(values),
                          fname, append=True)
        else:
            with open(fname, 'w') as f:
                _, nSteps, hiddenSize = feature.size()
                for step in range(nSteps):
                    line = [startStep + step * stepSize] + \
                        feature[0, step, :].tolist()
                    line = [str(x) for x in line]
                    linestr = ' '.join(line) + '\n'
                    f.write(linestr)
    bar.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Build features for zerospeech \
                                      Track1 evaluation')
    parser.add_argument('pathDB', help='Path to the reference dataset')
    parser.add_argument('pathOut', help='Path to the output features')
    parser.add_argument('pathCheckpoint', help='Checkpoint to load')
    parser.add_argument('--extension', type=str, default='.wav')
    parser.add_argument('--addCriterion', action='store_true')
    parser.add_argument('--oneHot', action='store_true')
    parser.add_argument('--maxSizeSeq', default=64000, type=int)
    parser.add_argument('--train_mode', action='store_true')
    parser.add_argument('--format', default='fea', type=str,
                        choices=['npz', 'fea', 'npy', 'af'])
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--dimReduction', type=str, default=None)
    parser.add_argument('--centroidLimits', type=int, nargs=2, default=None)
    parser.add_argument('--getEncoded', action='store_true')
    parser.add_argument('--clusters', type=str, default=None)
    parser.add_argument('--seqNorm', action='store_true')

    args = parser.parse_args()

    if not os.path.isdir(args.pathOut):
        os.mkdir(args.pathOut)

    with open(os.path.join(os.path.dirname(args.pathOut),
                           f"{os.path.basename(args.pathOut)}.json"), 'w') \
            as file:
        json.dump(vars(args), file, indent=2)

    outData = [x[1] for x in
               findAllSeqs(args.pathDB, extension=args.extension,
                           loadCache=False)[0]]

    featureMaker = loadModel([args.pathCheckpoint])[0]
    stepSize = featureMaker.gEncoder.DOWNSAMPLING / 16000
    print(f"stepSize : {stepSize}")
    featureMaker = FeatureModule(featureMaker, args.getEncoded)
    featureMaker.collapse = False

    if args.addCriterion:
        criterion, nPhones = loadSupervisedCriterion(args.pathCheckpoint)
        featureMaker = ModelPhoneCombined(featureMaker, criterion,
                                          nPhones, args.oneHot)
    if args.dimReduction is not None:
        dimRed = loadDimReduction(args.dimReduction, args.centroidLimits)
        featureMaker = torch.nn.Sequential(featureMaker, dimRed)
    if args.clusters is not None:
        cluster_state_dict = torch.load(args.clusters)
        nClusters = cluster_state_dict['n_clusters']
        clusterModule = kMeanCluster(torch.zeros(1, nClusters,
                                                 cluster_state_dict["dim"]))
        clusterModule.load_state_dict(cluster_state_dict['state_dict'])
        mode = 'oneHot' if args.oneHot else 'softmax'
        print(f"{nClusters} clusters found")
        featureMaker = ModelClusterCombined(featureMaker, clusterModule,
                                            nClusters,
                                            mode).cuda()
    featureMaker = featureMaker.cuda(device=0)

    if not args.train_mode:
        featureMaker.eval()

    buildAllFeature(featureMaker, args.pathDB, args.pathOut,  outData,
                    stepSize=stepSize, strict=args.strict,
                    maxSizeSeq=args.maxSizeSeq,
                    format=args.format,
                    seqNorm=args.seqNorm)
