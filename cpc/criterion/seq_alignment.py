# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import progressbar
import torch
from multiprocessing import Lock, Manager, Process
from copy import deepcopy


def beam_search(score_preds, nKeep, blankLabel):

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


def NeedlemanWunschAlignScore(seq1, seq2, d, m, r, normalize=True):

    N1, N2 = len(seq1), len(seq2)

    # Fill up the errors
    tmpRes_ = [[None for x in range(N2 + 1)] for y in range(N1 + 1)]
    for i in range(N1 + 1):
        tmpRes_[i][0] = i * d
    for j in range(N2 + 1):
        tmpRes_[0][j] = j * d

    for i in range(N1):
        for j in range(N2):

            match = r if seq1[i] == seq2[j] else m
            v1 = tmpRes_[i][j] + match
            v2 = tmpRes_[i + 1][j] + d
            v3 = tmpRes_[i][j + 1] + d
            tmpRes_[i + 1][j + 1] = max(v1, max(v2, v3))

    i = j = 0
    res = -tmpRes_[N1][N2]
    if normalize:
        res /= float(N1)
    return res


def get_seq_PER(seqLabels, detectedLabels):
    return NeedlemanWunschAlignScore(seqLabels, detectedLabels, -1, -1, 0,
                                     normalize=True)


def getPER(dataLoader, featureMaker, blankLabel):

    bar = progressbar.ProgressBar(len(dataLoader))
    bar.start()

    out = 0
    n_items = 0
    n_keep_beam_search = 100
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
            preds = beam_search(output[rank],
                                n_keep_beam_search, blankLabel)[0][1]
            value = get_seq_PER(seqLabels, preds)
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
