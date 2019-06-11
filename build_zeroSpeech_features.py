import torchaudio
import os
import json
from train import findAllSeqs, loadModel
import torch

import argparse


def printProgressBar(iteration,
                     total,
                     prefix='',
                     suffix='',
                     decimals=1,
                     length=100,
                     fill='#'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent
                                  complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def buildFeature(featureMaker,
                 seqPath,
                 strict=False,
                 maxSizeSeq=64000):

    seq = torchaudio.load(seqPath)[0]
    sizeSeq = seq.size(1)
    start = 0
    out = []
    while start < sizeSeq:
        if strict and start + maxSizeSeq > sizeSeq:
            break
        end = min(sizeSeq, start + maxSizeSeq)
        subseq = (seq[:, start:end]).view(1, 1, -1).cuda()
        features, _, _ = featureMaker(subseq, None)
        out.append(features.detach().cpu())
        start += maxSizeSeq

    if strict and start < sizeSeq:
        subseq = (seq[:, -maxSizeSeq:]).view(1, 1, -1).cuda()
        features, _ = featureMaker(subseq)
        out.append(features[:, start:].detach().cpu())

    return torch.cat(out, dim=1)


def getArgs(pathCheckpoints):
    pathArgs = os.path.join(os.path.dirname(pathCheckpoints),
                            "checkpoint_args.json")
    with open(pathArgs, 'rb') as file:
        return json.load(file)


def buildAllFeature(featureMaker, pathDB, pathOut,
                    seqList, stepSize, strict=False,
                    maxSizeSeq=64000):

    totSeqs = len(seqList)
    startStep = stepSize / 2
    for nseq, seqPath in enumerate(seqList):
        printProgressBar(nseq, totSeqs)
        feature = buildFeature(featureMaker,
                               os.path.join(pathDB, seqPath),
                               strict=strict,
                               maxSizeSeq=maxSizeSeq)
        outName = os.path.basename(os.path.splitext(seqPath)[0]) + '.fea'

        with open(os.path.join(pathOut, outName), 'w') as file:

            _, nSteps, hiddenSize = feature.size()
            for step in range(nSteps):
                line = [startStep + step * stepSize] + \
                    feature[0, step, :].tolist()
                line = [str(x) for x in line]
                linestr = ' '.join(line) + '\n'
                file.write(linestr)

    printProgressBar(totSeqs, totSeqs)


def toOneHot(inputVector, nItems):

    batchSize, = inputVector.size()
    out = torch.zeros((batchSize, nItems), device=inputVector.device)
    out.scatter_(1, inputVector.view(-1, 1), 1)
    return out


class ModelCriterionCombined(torch.nn.Module):
    def __init__(self, model, criterion, nPhones, oneHot):
        super(ModelCriterionCombined, self).__init__()
        self.model = model
        self.criterion = criterion
        self.nPhones = nPhones
        self.oneHot = oneHot

    def forward(self, data, label):
        c_feature, encoded_data, _ = self.model(data, None)
        pred = self.criterion(c_feature, encoded_data, label)

        if self.oneHot:
            pred = pred.max(1)[1]
            pred = toOneHot(pred, self.nPhones)
        else:
            pred = torch.nn.functional.softmax(pred, dim=1)
        return pred.view(1, -1, self.nPhones), None, None


def loadCriterion(pathCheckpoint):
    from criterion import PhoneCriterion
    from train import parseSeqLabels, getCheckpointData

    _, _, args = getCheckpointData(os.path.dirname(pathCheckpoint))
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

    args = parser.parse_args()

    if not os.path.isdir(args.pathOut):
        os.mkdir(args.pathOut)

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

    params = {"strictFeatures": False, "MAX_SIZE_SEQ": 64000}
    modelList = []

    featureMaker = loadModel([args.pathCheckpoint])[0]

    if args.addCriterion:
        criterion, nPhones = loadCriterion(args.pathCheckpoint)
        featureMaker = ModelCriterionCombined(featureMaker, criterion,
                                              nPhones, args.oneHot)

    featureMaker = featureMaker.cuda()

    buildAllFeature(featureMaker, args.pathDB, args.pathOut,  outData, 0.01,
                    strict=params["strictFeatures"],maxSizeSeq=params["MAX_SIZE_SEQ"])
