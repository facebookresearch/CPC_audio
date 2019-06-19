import torchaudio
import os
import json
from train import loadModel
from dataset import findAllSeqs
import torch
import progressbar
import argparse
import numpy as np


def buildFeature(featureMaker, seqPath, strict=False, maxSizeSeq=64000):

    seq = torchaudio.load(seqPath)[0]
    sizeSeq = seq.size(1)
    start = 0
    out = []
    while start < sizeSeq:
        if strict and start + maxSizeSeq > sizeSeq:
            break
        end = min(sizeSeq, start + maxSizeSeq)
        subseq = (seq[:, start:end]).view(1, 1, -1).cuda()
        with torch.no_grad():
            features, _, _ = featureMaker(subseq, None)
        out.append(features.detach().cpu())
        start += maxSizeSeq

    if strict and start < sizeSeq:
        subseq = (seq[:, -maxSizeSeq:]).view(1, 1, -1).cuda()
        with torch.no_grad():
            features, _ = featureMaker(subseq)
        out.append(features[:, start:].detach().cpu())

    return torch.cat(out, dim=1)


def getArgs(pathCheckpoints):
    pathArgs = os.path.join(os.path.dirname(pathCheckpoints),
                            "checkpoint_args.json")
    with open(pathArgs, 'rb') as file:
        return json.load(file)


def buildAllFeature(featureMaker, pathDB, pathOut,
                    seqList, stepSize=0.01, strict=False,
                    maxSizeSeq=64000, format='txt'):

    totSeqs = len(seqList)
    startStep = stepSize / 2
    bar = progressbar.ProgressBar(maxval=totSeqs)
    bar.start()
    for nseq, seqPath in enumerate(seqList):
        bar.update(nseq)
        feature = buildFeature(featureMaker,
                               os.path.join(pathDB, seqPath),
                               strict=strict,
                               maxSizeSeq=maxSizeSeq)

        _, nSteps, hiddenSize = feature.size()
        outName = os.path.basename(os.path.splitext(seqPath)[0]) + '.fea'
        fname = os.path.join(pathOut, outName)

        if format == 'npz':
            time = [startStep + step * stepSize for step in range(nSteps)]
            values = feature.squeeze(0).cpu().numpy()
            with open(fname, 'wb') as f:
                np.savez(f, time=time, features=values)
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


def toOneHot(inputVector, nItems):

    batchSize, = inputVector.size()
    out = torch.zeros((batchSize, nItems), device=inputVector.device)
    out.scatter_(1, inputVector.view(-1, 1), 1)
    return out


class ModelPhoneCombined(torch.nn.Module):
    def __init__(self, model, criterion, nPhones, oneHot):
        super(ModelPhoneCombined, self).__init__()
        self.model = model
        self.criterion = criterion
        self.nPhones = nPhones
        self.oneHot = oneHot

    def forward(self, data, label):
        c_feature, _, _ = self.model(data, label)
        pred = self.criterion.getPrediction(c_feature)

        if self.oneHot:
            pred = pred.argmax(dim=1)
            pred = toOneHot(pred, self.nPhones)
        else:
            pred = torch.nn.functional.softmax(pred, dim=1)
        return pred.view(1, -1, self.nPhones), None, None


def loadCriterion(pathCheckpoint):
    from criterion import PhoneCriterion
    from train import parseSeqLabels, getCheckpointData

    *_, args = getCheckpointData(os.path.dirname(pathCheckpoint))
    _, nPhones = parseSeqLabels(args["pathPhone"])
    criterion = PhoneCriterion(args["hiddenGar"], nPhones)

    state_dict = torch.load(pathCheckpoint)
    criterion.load_state_dict(state_dict["cpcCriterion"])
    return criterion, nPhones


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Build features for zerospeech \
                                      Track1 evaluation')
    parser.add_argument('pathDB', help='Path to the reference dataset')
    parser.add_argument('pathOut', help='Path to the output features')
    parser.add_argument('pathCheckpoint', help='Checkpoint to load')
    parser.add_argument('--seqList', help='Sequences to analyze',
                        type=str, default=None)
    parser.add_argument('--recursionLevel', type=int, default=1)
    parser.add_argument('--addCriterion', action='store_true')
    parser.add_argument('--oneHot', action='store_true')
    parser.add_argument('--maxSizeSeq', default=64000, type=int)
    parser.add_argument('--format', default='txt', type=str, choices=['npz', 'txt'])

    args = parser.parse_args()

    if not os.path.isdir(args.pathOut):
        os.mkdir(args.pathOut)

    with open(os.path.join(os.path.dirname(args.pathOut),
                           f"{os.path.basename(args.pathOut)}.json"), 'w') \
            as file:
        json.dump(vars(args), file, indent=2)

    seqList = [x[1] for x in
               findAllSeqs(args.pathDB, extension='.wav',
                           recursionLevel=args.recursionLevel)[0]]
    itemList = [(os.path.splitext(os.path.basename(x))[0], x) for x in seqList]

    if args.seqList is None:
        outData = [f[1] for f in itemList]
    else:
        outData = []
        with open(args.seqList, 'r') as file:
            filterNames = [x.replace('\n', '') for x in file]

        itemList.sort()
        seqList.sort()
        indexSeqList = 0
        for index, data in enumerate(itemList):
            item, value = data
            while indexSeqList < len(filterNames) and \
                    filterNames[indexSeqList] < item:
                indexSeqList += 1

            if indexSeqList < len(filterNames) and \
                    filterNames[indexSeqList] == item:
                outData.append(value)

    featureMaker = loadModel([args.pathCheckpoint])[0]

    if args.addCriterion:
        criterion, nPhones = loadCriterion(args.pathCheckpoint)
        featureMaker = ModelPhoneCombined(featureMaker, criterion,
                                          nPhones, args.oneHot)

    featureMaker = featureMaker.cuda()
    featureMaker.eval()

    buildAllFeature(featureMaker, args.pathDB, args.pathOut,  outData,
                    stepSize=0.01, strict=False, maxSizeSeq=args.maxSizeSeq, format=args.format)
