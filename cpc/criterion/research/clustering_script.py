# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import numpy as np
import time
import argparse
import sys
import os
import json
from random import shuffle
from cpc.criterion.research.clustering import kMeanCluster, kMeanGPU, fastDPMean, \
    distanceEstimation
from cpc.criterion.research.dim_reduction import loadDimReduction


def getQuantile(sortedData, percent):
    return sortedData[int(percent * len(sortedData))]


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
    from cpc.feature_loader import loadModel, FeatureModule
    from cpc.dataset import findAllSeqs, filterSeqs, AudioBatchData

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
