import argparse
import math
import torchaudio
import os
import torch
import progressbar

PHONE_LIST = ['h#', 'epi', 'pau', 'd', 'g', 'p', 'b', 't', 'k', 'dx', 'q',
              'jh', 'ch', 's', 'sh', 'z',
              'zh', 'f', 'th', 'v', 'dh', 'm', 'n', 'ng', 'em', 'en', 'eng',
              'nx', 'l', 'r', 'w', 'y', 'hh', 'hv', 'el', 'iy', 'ih', 'eh',
              'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw',
              'ux', 'er', 'ax', 'ix', 'axr', 'ax-h', 'dcl', 'bcl', 'kcl',
              'gcl', 'pcl', 'tcl']

PHONE_REDUCTION_39 = {'ao': 'aa',
                      'ax': 'ah', 'ax-h': 'ah',
                      'axr': 'er',
                      'hv': 'hh',
                      'ix': 'ih',
                      'el': 'l',
                      'em': 'm',
                      'en': 'n', 'nx': 'n',
                      'zh': 'sh',
                      'ux': 'uw',
                      'pcl': 'h#', 'tcl': 'h#', 'kcl': 'h#', 'bcl': 'h#',
                      'dcl': 'h#', 'gcl': 'h#', 'pau': 'h#', 'epi': 'h#'}


PHONE_MATCH = {item: index for index, item in enumerate(PHONE_LIST)}


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def NeedlemanWunschAlignScore(seq1, seq2, d, m, r, normalize = True):

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


def getSequenceFromFile(pathFile, sequenceMatch):

    with open(pathFile, 'r') as file:
        lines = file.readlines()

    out = []
    for line in lines:
        data = line.split()
        label = data[-1]
        intLabel = sequenceMatch[label]
        out.append(intLabel)

    return out


def copyFile(pathIn, pathOut):
    os.system(f'cp {pathIn} {pathOut}')


def makeDIR(path):

    if not os.path.isdir(path):
        os.mkdir(path)


def getPhoneLabels(pathFile, sequenceMatch, step):
    with open(pathFile, 'r') as file:
        lines = file.readlines()

    out = []
    startIndex = 0

    for line in lines:
        data = line.split()
        _, end, label = data
        endIndex = int(end) // step
        out += [sequenceMatch[label] for x in range(startIndex, endIndex)]
        startIndex = endIndex

    return out


def savePhoneLabels(pathOut, labels):

    file = open(pathOut, 'w')
    for label in labels:
        file.write(' '.join([str(x) for x in label]))
        file.write("\n")
    file.close()


def transferDB(pathDB, pathOut, stepLabel, namePhone, phone_match, copyFiles):

    # /<CORPUS>/<USAGE>/<DIALECT>/<SEX><SPEAKER_ID>/<SENTENCE_ID>.<FILE_TYPE>
    SUFFIX_AUDIO = ".WAV"
    SUFFIX_PHONE = ".PHN"

    makeDIR(pathOut)
    dirDialects = [f for f in os.listdir(pathDB)
                   if os.path.isdir(os.path.join(pathDB, f))]

    fullLabelData = []

    for indexDialect, dialect in enumerate(dirDialects):

        outDialect = os.path.join(pathOut, str(indexDialect))
        makeDIR(outDialect)

        inDialect = os.path.join(pathDB, dialect)
        dirSpeakers = [f for f in os.listdir(inDialect)
                       if os.path.isdir(os.path.join(inDialect, f))]

        for speaker in dirSpeakers:
            outSpeaker = os.path.join(outDialect, speaker)
            makeDIR(outSpeaker)
            inSpeaker = os.path.join(inDialect, speaker)

            sentences = [f for f in os.listdir(inSpeaker)
                         if os.path.splitext(f)[1] == SUFFIX_AUDIO]

            for sentence in sentences:

                pathInSentence = os.path.join(inSpeaker, sentence)
                pathPhone = f'{os.path.splitext(pathInSentence)[0]}{SUFFIX_PHONE}'
                phoneLabels = getPhoneLabels(pathPhone, phone_match, stepLabel)

                newName = f'{indexDialect}-{speaker}-{os.path.splitext(sentence)[0]}'
                phoneLabels = [newName] + phoneLabels
                fullLabelData.append(phoneLabels)
                pathOutSentence = os.path.join(outSpeaker, f'{newName}.wav')

                if copyFiles:
                    copyFile(pathInSentence, pathOutSentence)

    pathPhoneLabels = os.path.join(pathOut, f"{namePhone}.txt")
    savePhoneLabels(pathPhoneLabels, fullLabelData)


def buildTrainValSplit(pathDB,
                       pathOut,
                       shareTrain):

    from train import findAllSeqs
    from random import shuffle
    seqNames, _ = findAllSeqs(pathDB, recursionLevel=2, extension=".wav")
    shuffle(seqNames)

    limit = int(len(seqNames) * 0.8)
    seqTrain, seqVal = seqNames[:limit], seqNames[limit:]

    pathOut = os.path.splitext(pathOut)[0]
    pathOutTrain, pathOutVal = f'{pathOut}_train.txt', f'{pathOut}_val.txt'

    with open(pathOutTrain, 'w') as file:
        for _, name in seqTrain:
            file.write(f'{os.path.splitext(os.path.basename(name))[0]}\n')
    with open(pathOutVal, 'w') as file:
        for _, name in seqVal:
            file.write(f'{os.path.splitext(os.path.basename(name))[0]}\n')


def getSequence(model, phoneCriterion, pathSeq, MAX_SIZE_SEQ=640000):

    seq = torchaudio.load(pathSeq)[0]
    sizeSeq = seq.size(1)
    start, out = 0, []
    while start < sizeSeq:
        end = min(sizeSeq, start + MAX_SIZE_SEQ)
        subseq = (seq[:, start:end]).view(1, 1, -1).cuda()
        cFeature, encodedData, _ = model(subseq, None)
        preds = phoneCriterion.getPrediction(cFeature)
        preds = preds.argmax(dim=1)
        out += [int(i) for i in preds]

        start += MAX_SIZE_SEQ

    return out


def collapseSeq(inSeq):

    out = [inSeq[0]]
    for item in inSeq:
        if item != out[-1]:
            out.append(item)

    return out


def getSeqPER(seqLabels, detectedLabels):
    return NeedlemanWunschAlignScore(seqLabels, detectedLabels, -1, -1, 0,
                                    normalize=True)


def getPER(model, phoneCriterion, seqList, collapse, phoneLabels):

    avgPER = stdPER = 0
    N = len(seqList)

    n_ = 0
    bar = progressbar.ProgressBar(maxval=N)
    bar.start()

    for index, sequence in enumerate(seqList):

        bar.update(index)

        baseName = os.path.basename(os.path.splitext(sequence)[0])
        seqLabels = phoneLabels[baseName]
        detectedLabels = getSequence(model, phoneCriterion, sequence)

        if collapse:
            seqLabels = collapseSeq(seqLabels)
            detectedLabels = collapseSeq(detectedLabels)

        PER = NeedlemanWunschAlignScore(seqLabels, detectedLabels, -1, -1, 0,
                                        normalize=False)
        avgPER += PER
        stdPER += PER*PER
        n_ += len(seqLabels)

    bar.finish()
    avgPER /= n_
    stdPER = math.sqrt(max(0, stdPER / n_ - avgPER*avgPER))
    return avgPER, stdPER


def getSeqList(pathDB, extension):

    dirDialects = [f for f in os.listdir(pathDB)
                   if os.path.isdir(os.path.join(pathDB, f))]
    output = []
    for dialect in dirDialects:
        pathDialect = os.path.join(pathDB, dialect)
        dirSpeakers = [f for f in os.listdir(pathDialect)
                       if os.path.isdir(os.path.join(pathDialect, f))]

        for speaker in dirSpeakers:
            pathSpeaker = os.path.join(pathDialect, speaker)
            seqs = [f for f in os.listdir(pathSpeaker)
                    if os.path.splitext(f)[1] == extension]
            output += [os.path.join(pathSpeaker, f) for f in seqs]

    return output


def getCPCModel(pathModel, nPhones):

    cdata = getCheckpointData(os.path.dirname(pathModel))
    if cdata is None:
        print(f"No data found for checkpoint {pathModel}")
        raise RuntimeError()

    _, _, locArgs = cdata
    model = loadModel([pathModel])[0]
    criterion = PhoneCriterion(locArgs.hiddenGar, nPhones, locArgs.onEncoder, locArgs.nItemsSupervised)
    state_dict = torch.load(pathModel)
    criterion.load_state_dict(state_dict["cpcCriterion"])

    return model, criterion


if __name__ == "__main__":
    from criterion import PhoneCriterion
    from train import loadModel

    parser = argparse.ArgumentParser(description='Trainer')
    subparsers = parser.add_subparsers(dest="command")

    parser_transfer = subparsers.add_parser("transfer")
    parser_transfer.add_argument("pathDB", type=str)
    parser_transfer.add_argument("pathOut", type=str)
    parser_transfer.add_argument("--noCopy", action='store_true')
    parser_transfer.add_argument("--phoneReduction", action='store_true')
    parser_transfer.add_argument(
        "--namePhone", default="phoneLabels", type=str)

    parser_transfer = subparsers.add_parser("split")
    parser_transfer.add_argument("pathDB", type=str)
    parser_transfer.add_argument("--shareTrain", type=float, default=0.8)

    parser_transfer = subparsers.add_parser("PER")
    parser_transfer.add_argument("pathDB", type=str)
    parser_transfer.add_argument("pathPhone", type=str)
    parser_transfer.add_argument("pathModel", type=str)
    parser_transfer.add_argument("--collapse", action='store_true')

    args = parser.parse_args()

    if args.command == "transfer":
        phone_match = PHONE_MATCH
        if args.phoneReduction:
            phone_match = {}
            currIndex = 0
            for value in PHONE_LIST:
                if value not in PHONE_REDUCTION_39:
                    phone_match[value] = currIndex
                    currIndex += 1
            for key, value in PHONE_REDUCTION_39.items():
                phone_match[key] = phone_match[value]

        print(phone_match, currIndex)
        transferDB(args.pathDB, args.pathOut, 160,
                   args.namePhone, phone_match, not args.noCopy)
    elif args.command == "split":
        buildTrainValSplit(args.pathDB, os.path.join(args.pathDB, "split"),
                           args.shareTrain)
    elif args.command == "PER":
        from train import getCheckpointData, parseSeqLabels

        phoneLabels, nPhones = parseSeqLabels(args.pathPhone)

        seqList = getSeqList(args.pathDB, ".wav")
        model, criterion = getCPCModel(args.pathModel, nPhones)

        model.cuda()
        model.eval()

        criterion.cuda()
        criterion.eval()
        avgPER, stdPER = getPER(model, criterion, seqList,
                                args.collapse, phoneLabels)

        print(f'Average PER {avgPER}')
        print(f'STD PER {stdPER}')
