import os
import random
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, BatchSampler

import torchaudio


class AudioBatchData(Dataset):

    # Work on this and on the sampler
    def __init__(self,
                 path,
                 sizeWindow,
                 seqNames,
                 phoneLabels):
        """
        Args:
            path (string): path to the training dataset
            sizeWindow (int): size of the sliding window
            seqNames (list): sequences to load
        """

        self.dbPath = path
        self.sizeWindow = sizeWindow
        self.loadAll(seqNames, phoneLabels)

    def parseSeqName(name):
        speaker, chapter, id = name.split('-')
        return speaker, chapter, id

    def loadAll(self, seqNames, phoneLabels):

        # Speakers
        self.speakers = []

        # Labels
        self.speakerLabel = []
        self.seqLabel = [0]
        self.phoneLabels = []
        speakerSize = 0
        self.phoneSize = 0 if phoneLabels is None else phoneLabels["step"]
        self.phoneStep = 0 if phoneLabels is None else \
            self.sizeWindow // self.phoneSize

        # Data
        self.data = []

        # To accelerate the process a bit
        seqNames.sort()

        for seq in seqNames:
            seqName = os.path.splitext(seq)[0]
            speaker, chapter, id = \
                AudioBatchData.parseSeqName(seqName)

            if len(self.speakers) == 0 or self.speakers[-1] != speaker:
                self.speakers.append(speaker)
                self.speakerLabel.append(speakerSize)

            fullPath = os.path.join(self.dbPath,
                                    os.path.join(speaker,
                                                 os.path.join(chapter, seq)))

            seq = torchaudio.load(fullPath)[0].view(-1)

            if phoneLabels is not None:
                for data in phoneLabels[seqName]:
                    self.phoneLabels.append(data)
                newSize = len(phoneLabels[seqName]) * (self.phoneSize)
                assert(seq.size(0) >= newSize)
                seq = seq[:newSize]

            sizeSeq = seq.size(0)

            self.data.append(seq)
            self.seqLabel.append(self.seqLabel[-1] + sizeSeq)
            speakerSize += sizeSeq

        self.speakerLabel.append(speakerSize)
        self.data = torch.cat(self.data, dim=0)

    def getPhonem(self, idx):
        idPhone = idx // self.phoneSize
        return self.phoneLabels[idPhone:(idPhone + self.phoneStep)]

    def getSpeakerLabel(self, idx):

        idSpeaker = next(x[0] for x in enumerate(
            self.speakerLabel) if x[1] > idx) - 1
        return idSpeaker

    def __len__(self):
        return len(self.data) // self.sizeWindow

    def __getitem__(self, idx):

        if self.phoneSize > 0:
            label = torch.tensor(self.getPhonem(idx), dtype=torch.long)
        else:
            label = torch.tensor(
                self.getSpeakerLabel(idx), dtype=torch.long)

        return self.data[idx:(self.sizeWindow
                              + idx)].view(1, -1), label

    def getNSpeakers(self):
        return len(self.speakers)

    def getNSeqs(self):
        return len(self.seqLabel) - 1

    def getSampler(self, batchSize, groupSize, type, offset):
        if type == "speaker":
            return AudioBatchSampler(batchSize, groupSize,
                                     self.speakerLabel, self.sizeWindow, offset)
        if type == "sequence":
            return AudioBatchSampler(batchSize, groupSize,
                                     self.seqLabel, self.sizeWindow, offset)
        sampler = RandomAudioSampler(len(self.data), self.sizeWindow, offset)
        return BatchSampler(sampler, batchSize, True)


class RandomAudioSampler(Sampler):

    def __init__(self,
                 dataSize,
                 sizeWindow,
                 offset):

        self.len = dataSize // sizeWindow
        self.sizeWindow = sizeWindow
        self.offset = offset

    def __iter__(self):
        offset = random.randint(0, self.sizeWindow // 2) if self.offset else 0
        return iter((offset + self.sizeWindow * torch.randperm(self.len)).tolist())

    def __len__(self):
        return self.len


class AudioBatchSampler(Sampler):
    r"""
    A batch sampler producing mini-batch where items can be divided in groups
    of same label. At each iteration, the sampler will return a vector of
    indices:
    [a1, a2, .., ak, b1, ..., bk, ...]

    Where the dataset elements ai share the same label a.

    Note:
        - you can have several groups with the same label in the same minibatch
        (because input labels are not necessary envenly represented)
        - if batchSize % k != 0 then the last group will have the size
        batchSize % k
        - if not @param strict when there is not enough sample in a label to
        make a group of k elements, all remaining elements in that label are
        taken
    """

    def __init__(self,
                 batchSize,
                 groupSize,             # k
                 samplingIntervals,     # ex: AudioBatchData.speakerLabel
                 sizeWindow,            # see AudioBatchData.sizeWindow
                 offset):               # (bool) random offset ?

        self.samplingIntervals = samplingIntervals
        self.sizeWindow = sizeWindow

        if self.samplingIntervals[0] != 0:
            raise AttributeError("Sampling intervals should start at zero")

        nWindows = len(self.samplingIntervals) - 1
        self.sizeSamplers = [(self.samplingIntervals[i+1] -
                              self.samplingIntervals[i]) // self.sizeWindow
                             for i in range(nWindows)]
        self.batchSize = batchSize
        self.groupSize = groupSize
        self.offset = offset
        if offset:
            self.sizeSamplers = [x - 1 for x in self.sizeSamplers]

    def __iter__(self):
        batch = []
        order = [[x, i] for i, x in enumerate(self.sizeSamplers)]
        samplers = [torch.randperm(s) for s in self.sizeSamplers]
        offset = random.randint(0, self.sizeWindow // 2) if self.offset else 0
        nSamplers = len(order) - 1
        while nSamplers >= 0:
            w = random.randint(0, nSamplers)
            shift = min(self.groupSize, order[w][0])
            shift = min(shift, self.batchSize - len(batch))
            indexSampler, nInterval = order[w]
            for p in range(shift):
                indexInInterval = samplers[nInterval][
                    self.sizeSamplers[nInterval]-indexSampler].item()
                batch.append(self.getIndex(indexInInterval, nInterval, offset))
                indexSampler += 1
            order[w][0] -= shift
            if order[w][0] == 0:
                del order[w]
                nSamplers -= 1
            if len(batch) == self.batchSize:
                yield batch
                batch = []
                order.sort(reverse=True)
                w = 0

    def getSpeakerMaxSize(self, idx):
        return self.sizeSamplers[idx]

    def getIndex(self, x, iInterval, offset):
        return offset + x * self.sizeWindow + self.samplingIntervals[iInterval]

    def __len__(self):
        return sum(self.sizeSamplers) // self.batchSize
