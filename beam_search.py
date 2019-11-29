import torch
import argparse
from random import shuffle
from feature_maker import FeatureModule, ModelPhoneCombined, \
    loadCriterion
from train import loadModel, parseSeqLabels
from dataset import findAllSeqs, filterSeqs, AudioBatchData
from criterion.seq_alignment import getPER

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Evaluate the Phone error rate of a model using Beam \
                    Search predictions')
    parser.add_argument('pathCheckpoint', help='Checkpoint to load')
    parser.add_argument('--pathDB', type=str,
                        default="/datasets01_101/LibriSpeech/022219/train-clean-100/")
    parser.add_argument('--pathPhone', type=str,
                        default="/private/home/mriviere/LibriSpeech/LibriSpeech100_labels_split/converted_aligned_phones.txt")
    parser.add_argument('--seqList', type=str, default=None)
    parser.add_argument('--sizeWindow', default=20480, type=int)
    parser.add_argument('--recursionLevel', default=2, type=int)
    parser.add_argument('--extension', default=".flac", type=str)
    parser.add_argument('--batchSizeGPU', default=8, type=int)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    nGPU = torch.cuda.device_count()
    batchSize = args.batchSizeGPU * nGPU

    featureMaker = loadModel([args.pathCheckpoint])[0]
    featureMaker = FeatureModule(featureMaker, False)
    featureMaker.collapse = False

    criterion, nPhones = loadCriterion(args.pathCheckpoint)
    featureMaker = ModelPhoneCombined(featureMaker, criterion, False)
    featureMaker.cuda()
    featureMaker = torch.nn.DataParallel(featureMaker)
    featureMaker.eval()

    seqNames, speakers = findAllSeqs(args.pathDB,
                                     recursionLevel=args.recursionLevel,
                                     extension=args.extension)

    if args.seqList is not None:
        seqNames = filterSeqs(args.seqList, seqNames)
    if args.debug:
        shuffle(seqNames)
        seqNames = seqNames[:100]

    print("Loading the phone labels at " + args.pathPhone)
    phoneLabels, nPhones = parseSeqLabels(args.pathPhone)
    print(f"{nPhones} phones")
    dataset = AudioBatchData(args.pathDB,
                             args.sizeWindow,
                             seqNames,
                             phoneLabels,
                             list(speakers))

    trainLoader = dataset.getDataLoader(batchSize, "sequential",
                                        False, numWorkers=0)

    PER = getPER(trainLoader, featureMaker, nPhones)
    print(f"PER: {PER}, acc {1.0 - PER}")
