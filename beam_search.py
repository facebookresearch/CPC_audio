from copy import deepcopy
import torch
import argparse
from multiprocessing import Lock, Manager, Process
from eval_PER import getSeqPER
from random import shuffle

import progressbar


def beamSearch(score_preds, nKeep, blankLabel):

    T, P = score_preds.shape
    beams = set([''])
    pb_t_1 = {"": 1}
    pnb_t_1 = {"": 0}

    def getLastNumber(b):
        return int(b.split(',')[-1])

    for t in range(T):

        nextBeams = set()
        pb_t = {}
        pnb_t = {}
        for i_beam, b in enumerate(beams):
            if b not in pb_t:
                pb_t[b] = 0
                pnb_t[b] = 0

            if len(b) > 0:
                pnb_t[b] += pnb_t_1[b] * score_preds[t, getLastNumber(b)]
            pb_t[b] = (pnb_t_1[b] + pb_t_1[b]) * score_preds[t, blankLabel]
            nextBeams.add(b)

            for c in range(P):
                if c == blankLabel:
                    continue

                b_ = b + "," + str(c)
                if b_ not in pb_t:
                    pb_t[b_] = 0
                    pnb_t[b_] = 0

                if b != "" and getLastNumber(b) == c:
                    pnb_t[b_] += pb_t_1[b] * score_preds[t, c]
                else:
                    pnb_t[b_] += (pb_t_1[b] + pnb_t_1[b]) * score_preds[t, c]
                nextBeams.add(b_)

        allPreds = [(pb_t[b] + pnb_t[b], b) for b in nextBeams]
        allPreds.sort(reverse=True)

        beams = [x[1] for x in allPreds[:nKeep]]
        pb_t_1 = deepcopy(pb_t)
        pnb_t_1 = deepcopy(pnb_t)

    output = []
    for score, x in allPreds[:nKeep]:
        output.append((score, [int(y) for y in x.split(',') if len(y) > 0]))
    return output


def collapseLabelChain(inputLabels):

    # Shape N,T
    N, T = inputLabels.size()
    outSizes = torch.zeros(N, device=inputLabels.device, dtype=torch.int64)
    output = []
    for l in range(N):
        status = inputLabels[l, :-1] - inputLabels[l, 1:]
        status = torch.cat([torch.ones(1, device=status.device,
                                       dtype=status.dtype),
                            status], dim=0)
        outSizes[l] = (status != 0).sum()
        output.append(inputLabels[l][status != 0])
    maxSize = int(outSizes.max().item())
    paddedOutput = torch.zeros(N, maxSize,
                               device=inputLabels.device,
                               dtype=torch.int64)

    for l in range(N):
        S = int(outSizes[l])
        paddedOutput[l, :S] = output[l]

    return paddedOutput, outSizes


def getPER(dataLoader, featureMaker, blankLabel):

    bar = progressbar.ProgressBar(len(dataLoader))
    bar.start()

    out = 0
    n_items = 0
    for index, data in enumerate(dataLoader):

        bar.update(index)
        with torch.no_grad():
            output = featureMaker(data).cpu().numpy()
        labels = data[1]
        labels, targetSize = collapseLabelChain(labels)
        lock = Lock()

        def per(rank, outScore):
            S = int(targetSize[rank])
            seqLabels = labels[rank, :S]
            preds = beamSearch(output[rank],
                               100, blankLabel)[0][1]
            value = getSeqPER(seqLabels, preds)
            with lock:
                outScore.value += value

        manager = Manager()
        outScore = manager.Value('f', 0.)

        N, S, D = output.shape
        processes = []
        for rank in range(N):
            p = Process(
                target=per, args=(rank, outScore))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        out += outScore.value
        n_items += N

    bar.finish()
    return (out / n_items)


if __name__ == "__main__":
    from feature_maker import FeatureModule, ModelPhoneCombined, \
        loadCriterion
    from train import loadModel, parseSeqLabels
    from dataset import findAllSeqs, filterSeqs, AudioBatchData

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
