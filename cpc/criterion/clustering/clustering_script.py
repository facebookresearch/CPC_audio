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
from cpc.criterion.clustering import kMeanCluster, kMeanGPU
from pathlib import Path


def getQuantile(sortedData, percent):
    return sortedData[int(percent * len(sortedData))]


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Clustering module using kmeans or dpmeans.')
    parser.add_argument('pathCheckpoint', type=str,
                        help="Path to the checkpoint of CPC module.")
    parser.add_argument('pathOutput', type=str,
                        help="Path to the output clustering checkpoint.")
    parser.add_argument(
        '--pathDB', type=str,
        default="/datasets01/LibriSpeech/022219/train-clean-100/")
    parser.add_argument('-k', '--nClusters', type=int, default=50,
                        help="Number of clusters for kmeans algorithm (default: 50).")
    parser.add_argument('-g',  '--nGroups', type=int, default=1,
                        help="Number of groups for kmeans algorithm (default: 1).")
    parser.add_argument('-n', '--MAX_ITER', type=int, default=100,
                        help="Number of iterations (default: 150).")
    parser.add_argument('--recursionLevel', type=int, default=2,
                        help="The speaker recursionLevel in the training dataset (default: 2).")
    parser.add_argument('--extension', type=str, default='.flac',
                        help="The audio file extension (default: .flac).")
    parser.add_argument('--seqList', type=str, default=None,
                        help="Specific the training sequence list (default: None).")
    parser.add_argument('--sizeWindow', type=int, default=10240,
                        help="The size of the window when loading audio data (default: 10240).")
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode, only use a small number of training data.')
    parser.add_argument('--encoder_layer', action='store_true',
                        help='Whether to use the output of the encoder for the clustering.')
    parser.add_argument('--level_gru', type=int, default=None,
                        help='Specify the LSTM hidden level to take the representation (default: None).')
    parser.add_argument('--batchSizeGPU', type=int, default=50,
                        help='Batch size of each GPU (default: 50).')
    parser.add_argument('--DPMean', action='store_true',
                        help='Activate DPMeans training instead of Kmeans.')
    parser.add_argument('-l', '--DPLambda', type=float, default=11,
                        help='Lambda parameter of DPMeans algo (default: 11).')
    parser.add_argument('--perIterSize', type=int, default=-1,
                        help='(Depreciated) Number of items per iteration (default: -1).')
    parser.add_argument('--train_mode', action='store_true',
                        help='Activate training CPC module too.')
    parser.add_argument('--dimReduction', type=str, default=None,
                        help='Dimentionality reduction (default: None)')
    parser.add_argument('--centroidLimits', type=int, nargs=2, default=None,
                        help='centroidLimits when using dimentionality reduction (default: None)')
    parser.add_argument('--getDistanceEstimation', action='store_true',
                        help='Get distance estimation')
    parser.add_argument('--save', action='store_true',
                        help='Save the intermediate checkpoints. The checkpoints will'
                        'be saved in the same directory as the output.')
    parser.add_argument('--load', action='store_true',
                        help='Load the last checkpoint from the same directory as the output.')
    parser.add_argument('--save-last', type=int, default=5,
                        help='Number of last checkpoints to be saved (default: 5).')
    return parser.parse_args(argv)


if __name__ == "__main__":
    torch.cuda.empty_cache()

    import os
    from cpc.feature_loader import loadModel, FeatureModule
    from cpc.dataset import findAllSeqs, filterSeqs, AudioBatchData

    args = parseArgs(sys.argv[1:])
    # Export absolute paths for later use
    args.pathCheckpoint = os.path.abspath(args.pathCheckpoint)
    args.pathOutput = os.path.abspath(args.pathOutput)
    args.pathDB = os.path.abspath(args.pathDB)

    if not args.load: 
        assert os.path.exists(args.pathOutput) is False, \
            f"The output file {args.pathOutput} already exists, please check the option --load !"
        assert os.path.exists(os.path.join(os.path.dirname(args.pathOutput), "checkpoint_last.pt")) is False, \
            f"Found last_checkpoint.pt in the output directory, please check the option --load !"

    print(args)
    seqNames, speakers = findAllSeqs(args.pathDB,
                                     speaker_level=args.recursionLevel,
                                     extension=args.extension,
                                     loadCache=True)

    if args.seqList is not None:
        seqNames = filterSeqs(args.seqList, seqNames)
    if args.debug:
        nsamples=1000
        print(f"Debug mode activated, get only {nsamples} samples!")
        shuffle(seqNames)
        seqNames = seqNames[:nsamples]
    if args.getDistanceEstimation:
        shuffle(seqNames)
        seqNames = seqNames[:5000]

    print("")
    print(f'Loading audio data at {args.pathDB}')
    start_time = time.time()
    dataset = AudioBatchData(args.pathDB,
                             args.sizeWindow,
                             seqNames,
                             None,
                             len(speakers))
    print(f"Dataset loaded in {time.time()-start_time} seconds !")
    print("")

    nGPUs = torch.cuda.device_count()
    batchSize = args.batchSizeGPU * nGPUs
    trainLoader = dataset.getDataLoader(batchSize, "uniform",
                                        False, numWorkers=0)
    print(f"Length of dataLoader: {len(trainLoader)}")
    print("")

    if args.level_gru is None:
        updateConfig = None
    else:
        updateConfig = argparse.Namespace(nLevelsGRU=args.level_gru)

    model = loadModel([args.pathCheckpoint][0], updateConfig=updateConfig)[0]
    featureMaker = FeatureModule(model, args.encoder_layer)
    print("Checkpoint loaded!")
    print("")

    if not args.train_mode:
        featureMaker.eval()
    featureMaker.cuda()

    # Check if dir exists
    if not os.path.exists(os.path.dirname(args.pathOutput)) and os.path.dirname(args.pathOutput):
        Path(os.path.dirname(args.pathOutput)).mkdir(parents=True, exist_ok=True)

    pathConfig = f"{os.path.splitext(args.pathOutput)[0]}_args.json"
    with open(pathConfig, 'w') as file:
        json.dump(vars(args), file, indent=2)

    out_state_dict = {}
    print("Starting the clustering...")
    start_time = time.time()
    clusters = kMeanGPU(trainLoader, featureMaker, args.nClusters, args.nGroups,
                            perIterSize=args.perIterSize,
                            MAX_ITER=args.MAX_ITER,
                            save=args.save, load=args.load, 
                            save_dir=os.path.dirname(args.pathOutput),
                            save_last=args.save_last,
                            ).cpu()

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
