import os
import random
import time
import torch
from copy import deepcopy
from torch.multiprocessing import Lock, Manager
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler

import torchaudio


class AudioBatchData(Dataset):

    def __init__(self,
                 path,
                 sizeWindow,
                 seqNames,
                 phoneLabelsDict,
                 speakerList,
                 MAX_SIZE_LOADED=4000000000,
                 GROUP_SIZE_LOADED=2000):
        """
        Args:
            - path (string): path to the training dataset
            - sizeWindow (int): size of the sliding window
            - seqNames (list): sequences to load
            - phoneLabels (dictionnary): if not None, a dictionnary with the
                                         following entries

                                         "step": size of a labelled window
                                         "$SEQ_NAME": list of phonem labels for
                                         the sequence $SEQ_NAME
        """
        self.MAX_SIZE_LOADED = MAX_SIZE_LOADED
        self.GROUP_SIZE_LOADED = GROUP_SIZE_LOADED
        self.dbPath = path
        self.sizeWindow = sizeWindow
        self.seqNames = deepcopy(seqNames)
        self.prepare()
        self.speakers = deepcopy(speakerList)
        self.speakers.sort()
        self.data = []

        self.phoneSize = 0 if phoneLabelsDict is None else \
            phoneLabelsDict["step"]
        self.phoneStep = 0 if phoneLabelsDict is None else \
            self.sizeWindow // self.phoneSize

        self.phoneLabelsDict = deepcopy(phoneLabelsDict)
        self.loadNextPack()

    def splitSeqTags(seqName):
        path = os.path.normpath(seqName)
        return path.split(os.sep)

    def clear(self):
        if 'data' in self.__dict__:
            del self.data
        if 'speakerLabel' in self.__dict__:
            del self.speakerLabel
        if 'phoneLabels' in self.__dict__:
            del self.phoneLabels
        if 'seqLabel' in self.__dict__:
            del self.seqLabel

    def checkLength(self, item):
        _, seq = item
        info = torchaudio.info(os.path.join(self.dbPath, seq))[0]
        return info.length

    def prepare(self, poolSize=50):

        nSeqs = len(self.seqNames)
        random.shuffle(self.seqNames)
        start_time = time.time()

        # Data
        nprocess = min(50, nSeqs)
        sliceSize = nSeqs // nprocess
        mutex = Lock()

        def checkLength(rank, pool):
            indexStart = sliceSize * rank
            indexEnd = min(nSeqs, indexStart + sliceSize)
            packageSize, start = 0, indexStart
            output = []
            for index, item in enumerate(self.seqNames[indexStart:indexEnd]):
                _, seq = item
                l_ = torchaudio.info(os.path.join(self.dbPath, seq))[0].length
                packageSize += l_
                if packageSize > self.MAX_SIZE_LOADED:
                    output.append([start, index, packageSize])
                    packageSize, start = 0, index
            output.append([start, indexEnd, packageSize])
            mutex.acquire()
            pool += output
            mutex.release()

        processes = []
        manager = Manager()
        pool = manager.list()
        for rank in range(nprocess + 1):
            p = torch.multiprocessing.Process(
                target=checkLength, args=(rank, pool))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        pool.sort()
        self.packageIndex, self.totSize = [], 0
        currSize, start = 0, 0
        for item in pool:
            iStart, iEnd, size = item
            currSize += size
            self.totSize += size
            if currSize > self.MAX_SIZE_LOADED:
                self.packageIndex.append([start, iEnd])
                currSize, start = 0, iEnd
        self.packageIndex.append([start, iStart+currSize])
        print(f'Scanned {len(self.seqNames)} sequences '
              f'in {time.time() - start_time:.2f} seconds')
        self.currentPack = -1

    def getNPacks(self):
        return len(self.packageIndex)

    def loadNextPack(self):
        self.currentPack += 1
        self.currentPack = self.currentPack % len(self.packageIndex)
        seqStart, seqEnd = self.packageIndex[self.currentPack]
        self.clear()
        self.loadAll(self.seqNames[seqStart:seqEnd])

    def loadAll(self, seqNames):

        # Labels
        self.speakerLabel = [0]
        self.seqLabel = [0]
        self.phoneLabels = []
        speakerSize = 0
        indexSpeaker = 0

        # Data
        nprocess = min(50, len(seqNames))
        sliceSize = len(seqNames) // nprocess
        mutex = Lock()

        def load(index, pool):
            indexStart = sliceSize * index
            indexEnd = min(len(seqNames), indexStart + sliceSize)
            for index in range(indexStart, indexEnd):
                speaker, seq = seqNames[index]
                seqName = os.path.basename(os.path.splitext(seq)[0])
                fullPath = os.path.join(self.dbPath, seq)
                seq = torchaudio.load(fullPath)[0].view(-1)
                mutex.acquire()
                pool.append((speaker, seqName, seq))
                mutex.release()

        processes = []
        manager = Manager()
        pool = manager.list()
        for rank in range(nprocess):
            p = torch.multiprocessing.Process(target=load, args=(rank, pool))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # To accelerate the process a bit
        pool.sort()
        tmpData = []
        start_time = time.time()

        for speaker, seqName, seq in pool:
            while self.speakers[indexSpeaker] < speaker:
                indexSpeaker += 1
                self.speakerLabel.append(speakerSize)
            if self.speakers[indexSpeaker] != speaker:
                raise ValueError(f'{speaker} invalid speaker')

            if self.phoneLabelsDict is not None:
                for data in self.phoneLabelsDict[seqName]:
                    self.phoneLabels.append(data)
                newSize = len(self.phoneLabelsDict[seqName]) * self.phoneSize
                assert(seq.size(0) >= newSize)
                seq = seq[:newSize]

            sizeSeq = seq.size(0)
            tmpData.append(seq)
            self.seqLabel.append(self.seqLabel[-1] + sizeSeq)
            speakerSize += sizeSeq
            del seq

        self.speakerLabel.append(speakerSize)
        self.data = torch.cat(tmpData, dim=0)
        print(f'Loaded {len(seqNames)} sequences '
              f'in {time.time() - start_time:.2f} seconds')

    def getPhonem(self, idx):
        idPhone = idx // self.phoneSize
        return self.phoneLabels[idPhone:(idPhone + self.phoneStep)]

    def getSpeakerLabel(self, idx):
        idSpeaker = next(x[0] for x in enumerate(
            self.speakerLabel) if x[1] > idx) - 1
        return idSpeaker

    def __len__(self):
        return self.totSize // self.sizeWindow

    def __getitem__(self, idx):

        if idx < 0 or idx >= len(self.data) - self.sizeWindow - 1:
            print(idx)

        if self.phoneSize > 0:
            label = torch.tensor(self.getPhonem(idx), dtype=torch.long)
        else:
            label = torch.tensor(self.getSpeakerLabel(idx), dtype=torch.long)

        outData = self.data[idx:(self.sizeWindow + idx)].view(1, -1)
        return outData, label

    def getNSpeakers(self):
        return len(self.speakers)

    def getNSeqs(self):
        return len(self.seqLabel) - 1

    def getBaseSampler(self, type, batchSize, offset):
        if type == "samespeaker":
            return SameSpeakerSampler(batchSize, self.speakerLabel,
                                      self.sizeWindow, offset)
        if type == "samesequence":
            return SameSpeakerSampler(batchSize, self.seqLabel,
                                      self.sizeWindow, offset)
        if type == "sequential":
            return SequentialSampler(len(self.data), self.sizeWindow,
                                     offset, batchSize)
        sampler = UniformAudioSampler(len(self.data), self.sizeWindow,
                                      offset)
        return BatchSampler(sampler, batchSize, True)

    def getDataLoader(self, batchSize, type, randomOffset, numWorkers=0):
        r"""
        Get a batch sampler for the current dataset.
        Args:
            - batchSize (int): batch size
            - groupSize (int): in the case of type in ["speaker", "sequence"]
            number of items sharing a same label in the group
            (see AudioBatchSampler)
            - type (string):
                type == "speaker": grouped sampler speaker-wise
                type == "sequence": grouped sampler sequence-wise
                type == "sequential": sequential sampling
                else: uniform random sampling of the full audio
                vector
            - randomOffset (bool): if True add a random offset to the sampler
                                   at the begining of each iteration
        """
        def samplerCall():
            offset = random.randint(0, self.sizeWindow // 2) \
                if randomOffset else 0
            return self.getBaseSampler(type, batchSize, offset)

        return AudioLoader(self, samplerCall, len(self.packageIndex),
                           self.loadNextPack, self.__len__() // batchSize,
                           numWorkers)


class AudioLoader(object):

    def __init__(self,
                 dataset,
                 samplerCall,
                 nLoop,
                 updateCall,
                 size,
                 numWorkers):
        self.samplerCall = samplerCall
        self.updateCall = updateCall
        self.nLoop = nLoop
        self.size = size
        self.dataset = dataset
        self.numWorkers = numWorkers

    def __len__(self):
        return self.size

    def __iter__(self):

        for i in range(self.nLoop):
            sampler = self.samplerCall()
            dataloader = DataLoader(self.dataset,
                                    batch_sampler=sampler,
                                    num_workers=self.numWorkers)
            for x in dataloader:
                yield x
            if i < self.nLoop - 1:
                self.updateCall()


class UniformAudioSampler(Sampler):

    def __init__(self,
                 dataSize,
                 sizeWindow,
                 offset):

        self.len = dataSize // sizeWindow
        self.sizeWindow = sizeWindow
        self.offset = offset
        if self.offset > 0:
            self.len -= 1

    def __iter__(self):
        return iter((self.offset
                     + self.sizeWindow * torch.randperm(self.len)).tolist())

    def __len__(self):
        return self.len


class SequentialSampler(Sampler):

    def __init__(self, dataSize, sizeWindow, offset, batchSize):

        self.len = (dataSize // sizeWindow) // batchSize
        self.sizeWindow = sizeWindow
        self.offset = offset
        self.startBatches = [x * (dataSize // batchSize)
                             for x in range(batchSize)]
        self.batchSize = batchSize
        if self.offset > 0:
            self.len -= 1

    def __iter__(self):
        for idx in range(self.len):
            yield [self.offset + self.sizeWindow * idx
                   + start for start in self.startBatches]

    def __len__(self):
        return self.len


class SameSpeakerSampler(Sampler):

    def __init__(self,
                 batchSize,
                 samplingIntervals,
                 sizeWindow,
                 offset):

        self.samplingIntervals = samplingIntervals
        self.sizeWindow = sizeWindow
        self.batchSize = batchSize
        self.offset = offset

        if self.samplingIntervals[0] != 0:
            raise AttributeError("Sampling intervals should start at zero")

        nWindows = len(self.samplingIntervals) - 1
        self.sizeSamplers = [(self.samplingIntervals[i+1] -
                              self.samplingIntervals[i]) // self.sizeWindow
                             for i in range(nWindows)]

        if self.offset > 0:
            self.sizeSamplers = [max(0, x - 1) for x in self.sizeSamplers]

        order = [(x, torch.randperm(val).tolist())
                 for x, val in enumerate(self.sizeSamplers) if val > 0]

        # Build Batches
        self.batches = []
        for indexSampler, randperm in order:
            indexStart, sizeSampler = 0, self.sizeSamplers[indexSampler]
            while indexStart < sizeSampler:
                indexEnd = min(sizeSampler, indexStart + self.batchSize)
                locBatch = [self.getIndex(x, indexSampler)
                            for x in randperm[indexStart:indexEnd]]
                indexStart = indexEnd
                self.batches.append(locBatch)

    def __len__(self):
        return len(self.batches)

    def getIndex(self, x, iInterval):
        return self.offset + x * self.sizeWindow \
            + self.samplingIntervals[iInterval]

    def __iter__(self):
        random.shuffle(self.batches)
        return iter(self.batches)


def findAllSeqs(dirName,
                recursionLevel=2,
                extension='.flac'):

    dirName = os.path.join(dirName, '')
    dirList = [dirName]
    prefixSize = len(dirName)
    speakers = set([])

    for recursion in range(recursionLevel):
        nextList = []
        for item in dirList:
            nextList += [os.path.join(item, f) for f in os.listdir(item)
                         if os.path.isdir(os.path.join(item, f))]
        dirList = nextList

    outSequences = []
    for directory in dirList:
        basePath = directory[prefixSize:]
        try:
            speaker = int(os.path.normpath(basePath).split(os.sep)[0])
        except ValueError:
            speaker = 0
        speakers.add(speaker)
        for item in os.listdir(directory):
            if os.path.splitext(item)[1] != extension:
                continue
            outSequences.append((speaker, os.path.join(basePath, item)))

    return outSequences, speakers


def filterSeqs(pathTxt, seqCouples):
    with open(pathTxt, 'r') as f:
        inSeqs = [p.replace('\n', '') for p in f.readlines()]

    inSeqs.sort()
    seqCouples.sort(key=lambda x: x[1])
    output, index = [], 0
    for x in seqCouples:
        seq = os.path.basename(os.path.splitext(x[1])[0])
        while index < len(inSeqs) and seq > inSeqs[index]:
            index += 1
        if index == len(inSeqs):
            break
        if seq == inSeqs[index]:
            output.append(x)
    return output
