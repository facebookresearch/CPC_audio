# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import json
import argparse
import progressbar
from pathlib import Path
from random import shuffle
from time import time
import torch
from cpc.dataset import findAllSeqs
from cpc.feature_loader import (
    buildFeature,
    FeatureModule,
    loadModel,
)
from cpc.clustering.clustering import loadClusterModule


def readArgs(path_dir):
    print(f"Loading args from {path_dir}")
    pathArgs = Path(path_dir) / "args.json"
    with open(pathArgs, "r") as file:
        args = argparse.Namespace(**json.load(file))
    return args


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(
        description="Quantize audio files using CPC Clustering Module."
    )
    parser.add_argument(
        "pathCheckpoint", type=str, help="Path to the clustering checkpoint."
    )
    parser.add_argument(
        "pathDB", type=str, help="Path to the dataset that we want to quantize."
    )
    parser.add_argument("pathOutput", type=str, help="Path to the output directory.")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="If you want to divide the dataset in small splits, specify it "
        "with idxSplit-numSplits (idxSplit > 0), eg. --split 1-20.",
    )
    parser.add_argument(
        "--file_extension",
        type=str,
        default=".flac",
        help="Extension of the audio files in the dataset (default: .flac).",
    )
    parser.add_argument(
        "--max_size_seq",
        type=int,
        default=10240,
        help="Maximal number of frames to consider "
        "when computing a batch of features (defaut: 10240).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size used to compute features "
        "when computing each file (defaut: 8).",
    )
    parser.add_argument(
        "--strict",
        type=bool,
        default=True,
        help="If activated, each batch of feature "
        "will contain exactly max_size_seq frames (defaut: True).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Load only a very small amount of files for debugging purposes.",
    )
    parser.add_argument(
        "--nobatch",
        action="store_true",
        help="Don't use batch implementation of when building features."
        "NOTE: This can have better quantized units as we can set "
        "model.gAR.keepHidden = True (line 162), but the quantization"
        "will be a bit longer.",
    )
    parser.add_argument(
        "--recursionLevel",
        type=int,
        default=1,
        help="Speaker level in pathDB (defaut: 1). This is only helpful"
        "when --separate-speaker is activated.",
    )
    parser.add_argument(
        "--separate-speaker",
        action="store_true",
        help="Separate each speaker with a different output file.",
    )
    return parser.parse_args(argv)


def main(argv):
    # Args parser
    args = parseArgs(argv)

    print("=============================================================")
    print(f"Quantizing data from {args.pathDB}")
    print("=============================================================")

    # Check if directory exists
    if not os.path.exists(args.pathOutput):
        print("")
        print(f"Creating the output directory at {args.pathOutput}")
        Path(args.pathOutput).mkdir(parents=True, exist_ok=True)

    # Get splits
    if args.split:
        assert (
            len(args.split.split("-")) == 2
            and int(args.split.split("-")[1]) >= int(args.split.split("-")[0]) >= 1
        ), "SPLIT must be under the form idxSplit-numSplits (numSplits >= idxSplit >= 1), eg. --split 1-20"
        idx_split, num_splits = args.split.split("-")
        idx_split = int(idx_split)
        num_splits = int(num_splits)

    # Find all sequences
    print("")
    print(
        f"Looking for all {args.file_extension} files in {args.pathDB} with speakerLevel {args.recursionLevel}"
    )
    seqNames, speakers = findAllSeqs(
        args.pathDB,
        speaker_level=args.recursionLevel,
        extension=args.file_extension,
        loadCache=True,
    )
    print(f"Done! Found {len(seqNames)} files and {len(speakers)} speakers!")
    if args.separate_speaker:
        seqNames_by_speaker = {}
        for seq in seqNames:
            speaker = seq[1].split("/")[args.recursionLevel - 1]
            if speaker not in seqNames_by_speaker:
                seqNames_by_speaker[speaker] = []
            seqNames_by_speaker[speaker].append(seq)

    # Check if output file exists
    if not args.split:
        nameOutput = "quantized_outputs.txt"
    else:
        nameOutput = f"quantized_outputs_split_{idx_split}-{num_splits}.txt"
    if args.separate_speaker is False:
        outputFile = os.path.join(args.pathOutput, nameOutput)
        assert not os.path.exists(
            outputFile
        ), f"Output file {outputFile} already exists !!!"

    # Get splits
    if args.split:
        startIdx = len(seqNames) // num_splits * (idx_split - 1)
        if idx_split == num_splits:
            endIdx = len(seqNames)
        else:
            endIdx = min(len(seqNames) // num_splits * idx_split, len(seqNames))
        seqNames = seqNames[startIdx:endIdx]
        print("")
        print(
            f"Quantizing split {idx_split} out of {num_splits} splits, with {len(seqNames)} files (idx in range({startIdx}, {endIdx}))."
        )

    # Debug mode
    if args.debug:
        nsamples = 20
        print("")
        print(f"Debug mode activated, only load {nsamples} samples!")
        seqNames = seqNames[:nsamples]

    # Load Clustering args
    assert args.pathCheckpoint.endswith(".pt")
    clustering_args = readArgs(Path(args.pathCheckpoint).parent)
    print("")
    print(
        f"Clutering args:\n{json.dumps(vars(clustering_args), indent=4, sort_keys=True)}"
    )
    print("-" * 50)

    # Load CluterModule
    clusterModule = loadClusterModule(args.pathCheckpoint)
    clusterModule.cuda()

    # Load FeatureMaker
    print("")
    print("Loading CPC FeatureMaker")
    if "level_gru" in vars(clustering_args) and clustering_args.level_gru is not None:
        updateConfig = argparse.Namespace(nLevelsGRU=clustering_args.level_gru)
    else:
        updateConfig = None
    model = loadModel([clustering_args.pathCheckpoint], updateConfig=updateConfig)[0]
    ## If we don't apply batch implementation, we can set LSTM model to keep hidden units
    ## making the quality of the quantized units better
    if args.nobatch:
        model.gAR.keepHidden = True
    featureMaker = FeatureModule(model, clustering_args.encoder_layer)
    if clustering_args.dimReduction is not None:
        dimRed = loadDimReduction(
            clustering_args.dimReduction, clustering_args.centroidLimits
        )
        featureMaker = torch.nn.Sequential(featureMaker, dimRed)
    if not clustering_args.train_mode:
        featureMaker.eval()
    featureMaker.cuda()

    print("CPC FeatureMaker loaded!")

    # Quantization of files
    print("")
    print(f"Quantizing audio files...")
    seqQuantLines = []
    bar = progressbar.ProgressBar(maxval=len(seqNames))
    bar.start()
    start_time = time()
    for index, vals in enumerate(seqNames):
        bar.update(index)

        file_path = vals[1]
        file_path = os.path.join(args.pathDB, file_path)

        # Get features & quantizing
        cFeatures = buildFeature(
            featureMaker, file_path, seqNorm=False, strict=args.strict
        ).cuda()

        nGroups = cFeatures.size(-1) // clusterModule.Ck.size(-1)

        cFeatures = cFeatures.view(1, -1, clusterModule.Ck.size(-1))

        if len(vals) > 2 and int(vals[-1]) > 9400000:  # Librilight, to avoid OOM
            clusterModule = clusterModule.cpu()
            cFeatures = cFeatures.cpu()
            qFeatures = torch.argmin(clusterModule(cFeatures), dim=-1)
            clusterModule = clusterModule.cuda()
        else:
            qFeatures = torch.argmin(clusterModule(cFeatures), dim=-1)
        qFeatures = qFeatures[0].detach().cpu().numpy()

        # Transform to quantized line
        quantLine = ",".join(
            [
                "-".join([str(i) for i in item])
                for item in qFeatures.reshape(-1, nGroups)
            ]
        )
        seqQuantLines.append(quantLine)

    bar.finish()
    print(f"...done {len(seqQuantLines)} files in {time()-start_time} seconds.")

    # Saving outputs
    print("")
    print(f"Saving outputs to {outputFile}")
    outLines = []
    for vals, quantln in zip(seqNames, seqQuantLines):
        file_path = vals[1]
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        outLines.append("\t".join([file_name, quantln]))
    with open(outputFile, "w") as f:
        f.write("\n".join(outLines))


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
