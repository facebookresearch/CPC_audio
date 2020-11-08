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
from cpc.feature_loader import buildFeature, FeatureModule, loadModel, buildFeature_batch
from cpc.criterion.clustering import kMeanCluster

def readArgs(pathArgs):
    print(f"Loading args from {pathArgs}")
    with open(pathArgs, 'r') as file:
        args = argparse.Namespace(**json.load(file))
        
    return args

def writeArgs(pathArgs, args):
    with open(pathArgs, 'w') as file:
        json.dump(vars(args), file, indent=2)

def loadClusterModule(pathCheckpoint):
    print(f"Loading ClusterModule at {pathCheckpoint}")
    state_dict = torch.load(pathCheckpoint, map_location=torch.device('cpu'))
    clusterModule = kMeanCluster(torch.zeros(1, state_dict["n_clusters"], state_dict["dim"]))
    clusterModule.load_state_dict(state_dict["state_dict"])
    return clusterModule

def quantize_file(file_path, cpc_feature_function, clusterModule):
    # Get CPC features
    cFeatures = cpc_feature_function(file_path)
    if clusterModule.Ck.is_cuda:
        cFeatures = cFeatures.cuda()

    nGroups = cFeatures.size(-1)//clusterModule.Ck.size(-1) # groups information

    # Quantize the output of clustering on the CPC features
    cFeatures = cFeatures.view(1, -1, clusterModule.Ck.size(-1))
    if cFeatures.size(1) > 50000: # Librilight, to avoid GPU OOM, decrease when still OOM
        clusterModule = clusterModule.cpu()
        cFeatures = cFeatures.cpu()
        qFeatures = torch.argmin(clusterModule(cFeatures), dim=-1)
        if not args.cpu:
            clusterModule = clusterModule.cuda()
    else:
        qFeatures = torch.argmin(clusterModule(cFeatures), dim=-1)
    qFeatures = qFeatures[0].detach().cpu().numpy()

    # Transform to quantized line
    quantLine = ",".join(["-".join([str(i) for i in item]) for item in qFeatures.reshape(-1, nGroups)])

    return quantLine

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Quantize audio files using CPC Clustering Module.')
    parser.add_argument('pathClusteringCheckpoint', type=str,
                        help='Path to the clustering checkpoint.')
    parser.add_argument('pathDB', type=str,
                        help='Path to the dataset that we want to quantize.')
    parser.add_argument('pathOutputDir', type=str,
                        help='Path to the output directory.')
    parser.add_argument('--pathSeq', type=str,	
                       help='Path to the sequences (file names) to be included used '
                       '(if not speficied, included all files found in pathDB).')
    parser.add_argument('--split', type=str, default=None,
                        help='If you want to divide the dataset in small splits, specify it '
                        'with idxSplit-numSplits (idxSplit > 0), eg. --split 1-20.')
    parser.add_argument('--file_extension', type=str, default=".flac",
                          help="Extension of the audio files in the dataset (default: .flac).")
    parser.add_argument('--max_size_seq', type=int, default=10240,
                        help='Maximal number of frames to consider '
                        'when computing a batch of features (defaut: 10240).')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size used to compute features '
                        'when computing each file (defaut: 8).')
    parser.add_argument('--strict', type=bool, default=True,
                        help='If activated, each batch of feature '
                        'will contain exactly max_size_seq frames (defaut: True).')
    parser.add_argument('--debug', action='store_true',
                        help="Load only a very small amount of files for "
                        "debugging purposes.")
    parser.add_argument('--nobatch', action='store_true',
                        help="Don't use batch implementation of when building features."
                        "NOTE: This can have better quantized units as we can set "
                        "model.gAR.keepHidden = True (line 162), but the quantization"
                        "will be a bit longer.")
    parser.add_argument('--cpu', action='store_true',
                        help="Run on a cpu machine.")
    parser.add_argument('--resume', action='store_true',
                        help="Continue to quantize if an output file already exists.")
    return parser.parse_args(argv)

def main(argv):
    # Args parser
    args = parseArgs(argv)
    
    print("=============================================================")
    print(f"Quantizing data from {args.pathDB}")
    print("=============================================================")

    # Get splits
    if args.split:
        assert len(args.split.split("-"))==2 and int(args.split.split("-")[1]) >= int(args.split.split("-")[0]) >= 1, \
            "SPLIT must be under the form idxSplit-numSplits (numSplits >= idxSplit >= 1), eg. --split 1-20"
        idx_split, num_splits = args.split.split("-")
        idx_split = int(idx_split)
        num_splits = int(num_splits)

    # Find all sequences
    print("")
    print(f"Looking for all {args.file_extension} files in {args.pathDB}")
    seqNames, _ = findAllSeqs(args.pathDB,
                                 speaker_level=1,
                                 extension=args.file_extension,
                                 loadCache=True)
    if len(seqNames) == 0 or not os.path.splitext(seqNames[0][1])[1].endswith(args.file_extension):
        print(f"Seems like the _seq_cache.txt does not contain the correct extension, reload the file list")
        seqNames, _ = findAllSeqs(args.pathDB,
                                    speaker_level=1,
                                    extension=args.file_extension,
                                    loadCache=False)
    print(f"Done! Found {len(seqNames)} files!")

    # Filter specific sequences
    if args.pathSeq:
        print("")
        print(f"Filtering seqs in {args.pathSeq}")
        with open(args.pathSeq, 'r') as f:	
            seqs = set([x.strip() for x in f])	
        filtered = []	
        for s in seqNames:	
            if os.path.splitext(s[1].split('/')[-1])[0] in seqs:	
                filtered.append(s)	
        seqNames = filtered
        print(f"Done! {len(seqNames)} files filtered!")

    # Check if directory exists
    if not os.path.exists(args.pathOutputDir):
        print("")
        print(f"Creating the output directory at {args.pathOutputDir}")
        Path(args.pathOutputDir).mkdir(parents=True, exist_ok=True)
    writeArgs(os.path.join(args.pathOutputDir, "_info_args.json"), args)

    # Check if output file exists
    if not args.split:
        nameOutput = "quantized_outputs.txt"
    else:
        nameOutput = f"quantized_outputs_split_{idx_split}-{num_splits}.txt"
    outputFile = os.path.join(args.pathOutputDir, nameOutput)
    
    # Get splits
    if args.split:
        startIdx = len(seqNames) // num_splits * (idx_split-1)
        if idx_split == num_splits:
            endIdx = len(seqNames)
        else:
            endIdx = min(len(seqNames) // num_splits * idx_split, len(seqNames))
        seqNames = seqNames[startIdx:endIdx]
        print("")
        print(f"Quantizing split {idx_split} out of {num_splits} splits, with {len(seqNames)} files (idx in range({startIdx}, {endIdx})).")

    # Debug mode
    if args.debug:
        nsamples=20
        print("")
        print(f"Debug mode activated, only load {nsamples} samples!")
        # shuffle(seqNames)
        seqNames = seqNames[:nsamples]

    # Continue
    addEndLine = False # to add end line (\n) to first line or not
    if args.resume:
        if os.path.exists(outputFile):
            with open(outputFile, 'r') as f:
                lines = [line for line in f]
            existing_files = set([x.split()[0] for x in lines if x.split()])
            seqNames = [s for s in seqNames if os.path.splitext(s[1].split('/')[-1])[0] not in existing_files]
            print(f"Found existing output file, continue to quantize {len(seqNames)} audio files left!")
            if len(lines) > 0 and not lines[-1].endswith("\n"):
                addEndLine = True
    else:
        assert not os.path.exists(outputFile), \
            f"Output file {outputFile} already exists !!! If you want to continue quantizing audio files, please check the --resume option."

    assert len(seqNames) > 0, \
        "No file to be quantized!"

    # Load Clustering args
    assert args.pathClusteringCheckpoint[-3:] == ".pt"
    if os.path.exists(args.pathClusteringCheckpoint[:-3] + "_args.json"):
        pathConfig = args.pathClusteringCheckpoint[:-3] + "_args.json"
    elif os.path.exists(os.path.join(os.path.dirname(args.pathClusteringCheckpoint), "checkpoint_args.json")):
        pathConfig = os.path.join(os.path.dirname(args.pathClusteringCheckpoint), "checkpoint_args.json")
    else:
        assert False, \
            f"Args file not found in the directory {os.path.dirname(args.pathClusteringCheckpoint)}"
    clustering_args = readArgs(pathConfig)
    print("")
    print(f"Clutering args:\n{json.dumps(vars(clustering_args), indent=4, sort_keys=True)}")
    print('-' * 50)

    # Load CluterModule
    clusterModule = loadClusterModule(args.pathClusteringCheckpoint)
    if not args.cpu:
        clusterModule.cuda()

    # Load FeatureMaker
    print("")
    print("Loading CPC FeatureMaker")
    if not os.path.isabs(clustering_args.pathCheckpoint): # Maybe it's relative path
        clustering_args.pathCheckpoint = os.path.join(os.path.dirname(os.path.abspath(args.pathClusteringCheckpoint)), clustering_args.pathCheckpoint)
    assert os.path.exists(clustering_args.pathCheckpoint), \
        f"CPC path at {clustering_args.pathCheckpoint} does not exist!!"
    if 'level_gru' in vars(clustering_args) and clustering_args.level_gru is not None:
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
        dimRed = loadDimReduction(clustering_args.dimReduction, clustering_args.centroidLimits)
        featureMaker = torch.nn.Sequential(featureMaker, dimRed)
    if not clustering_args.train_mode:
        featureMaker.eval()
    if not args.cpu:
        featureMaker.cuda()
    def cpc_feature_function(x): 
        if args.nobatch is False:
            return buildFeature_batch(featureMaker, x,
                                                    seqNorm=False,
                                                    strict=args.strict,
                                                    maxSizeSeq=args.max_size_seq,
                                                    batch_size=args.batch_size)
        else:
             return buildFeature(featureMaker, x,
                                seqNorm=False,
                                strict=args.strict)
    print("CPC FeatureMaker loaded!")

    # Quantization of files
    print("")
    print(f"Quantizing audio files and saving outputs to {outputFile}...")
    f = open(outputFile, "a")
    bar = progressbar.ProgressBar(maxval=len(seqNames))
    bar.start()
    start_time = time()
    for index, vals in enumerate(seqNames):
        bar.update(index)

        file_path = vals[1]
        file_path = os.path.join(args.pathDB, file_path)

        # Quantizing
        quantLine = quantize_file(file_path, cpc_feature_function, clusterModule)

        # Save the outputs
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        outLine = "\t".join([file_name, quantLine])
        if addEndLine:
            f.write("\n"+outLine)
        else:
            f.write(outLine)
            addEndLine = True
    bar.finish()
    print(f"...done {len(seqNames)} files in {time()-start_time} seconds.")
    f.close()

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)

