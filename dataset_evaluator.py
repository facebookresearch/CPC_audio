import torch
import torchaudio

import os


def findAllSeq(dirName, recursionLevel=2, extension='.wav'):

    # Step one: directory recursion
    dirName = os.path.join(dirName, '')
    dirList = [dirName]
    prefixSize = len(dirName)

    for recursion in range(recursionLevel):
        nextList = []
        for item in dirList:
            nextList += [os.path.join(item, f) for f in os.listdir(item)
                         if os.path.isdir(os.path.join(item, f))]

        dirList = nextList
    outSequences = []
    for directory in dirList:
        basePath = directory[prefixSize:]
        for item in os.listdir(directory):
            if os.path.splitext(item)[1] != extension:
                continue
            outSequences.append(os.path.join(basePath, item))
    return outSequences


def splitSeqTags(seqName):
    path = os.path.normpath(seqName)
    return path.split(os.sep)


path = "/datasets01/LibriSpeech/022219/train-clean-100/"
coin = findAllSeq(path, extension='.flac')
coin = coin[:10]

sizes = []

for item in coin:
    info = torchaudio.info(os.path.join(path, item))
    sizes.append(info[0].length // 20480)

print(min(sizes))
print(max(sizes))
print(sum(sizes) / len(sizes))
