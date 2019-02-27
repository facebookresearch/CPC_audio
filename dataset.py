import os
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

import torchaudio


class AudioBatchData(Dataset):

    # Work on this and on the sampler
    def __init__(self,
                 path,
                 sizeWindow,
                 seqNames):
        """
        Args:
            path (string): path to the training dataset
            seqNamesPath (string): path to a file listing the sequences to load
        """

        self.dbPath = path
        self.sizeWindow = sizeWindow
        self.loadAll(seqNames)

    def parseSeqName(name):
        speaker, chapter, id = name.split('-')
        return speaker, chapter, id

    def loadAll(self, seqNames):

        # Speakers
        self.speakers = []

        # Labels
        self.speakerLabel = []
        self.seqLabel = [0]
        speakerSize = 0

        # Data
        self.data = []

        # To accelerate the process a bit
        seqNames.sort()

        for seq in seqNames:
            speaker, chapter, id = \
                AudioBatchData.parseSeqName(os.path.splitext(seq)[0])

            if len(self.speakers) == 0 or self.speakers[-1] != speaker:
                self.speakers.append(speaker)
                self.speakerLabel.append(speakerSize)

            fullPath = os.path.join(self.dbPath,
                                    os.path.join(speaker,
                                                 os.path.join(chapter, seq)))

            seq = torchaudio.load(fullPath)[0].view(-1)
            sizeSeq = seq.size(0)
            self.data.append(seq)
            self.seqLabel.append(self.seqLabel[-1] + sizeSeq)
            speakerSize += sizeSeq

        self.speakerLabel.append(speakerSize)
        self.data = torch.cat(self.data, dim=0)

    def getLabel(self, idx):

        idSpeaker = next(x[0] for x in enumerate(
            self.speakerLabel) if x[1] > idx) - 1
        return idSpeaker

    def __len__(self):
        return len(self.data) // self.sizeWindow

    def __getitem__(self, idx):

        speakerLabel = torch.tensor(
            self.getLabel(idx), dtype=torch.long)

        return self.data[idx:(self.sizeWindow
                         + idx)].view(1, -1), speakerLabel

    def getNSpeakers(self):
        return len(self.speakers)

    def getSpeakerOffset(self, x):
        return self.speakerLabel[x]

    def getSpeakerMaxSize(self, idx):
        return (self.speakerLabel[idx+1] - self.speakerLabel[idx]) \
                    // self.sizeWindow

    def getSampler(self, batchSize, groupSize):
        return AudioBatchSampler(batchSize, groupSize,
                                 self.speakerLabel, self.sizeWindow)

class AudioBatchSampler(Sampler):

    def __init__(self,
                 batchSize,
                 groupSize,
                 samplingIntervals,
                 sizeWindow):

        self.samplingIntervals = samplingIntervals
        self.sizeWindow = sizeWindow

        if self.samplingIntervals[0] != 0:
            raise AttributeError("Sampling intervals should start at zero")

        nWindows = len(self.samplingIntervals) -1
        self.sizeSamplers = [(self.samplingIntervals[i+1] -
                             self.samplingIntervals[i]) // self.sizeWindow
                             for i in range(nWindows)]
        self.samplers = [torch.randperm(s) for s in self.sizeSamplers]
        self.batchSize = batchSize
        self.groupSize = groupSize

    def __iter__(self):
        batch = []
        order = [[x, i] for i, x in enumerate(self.sizeSamplers)]
        order.sort(reverse = True)
        for idx in range(sum(self.sizeSamplers)):
            shift = min(self.groupSize, order[0][0])
            shift = min(shift, self.batchSize - len(batch))
            indexSampler, nInterval = order[0]
            for p in range(shift):
                itemIndexInInterval = self.samplers[nInterval][
                             self.sizeSamplers[nInterval]-indexSampler].item()
                batch.append(self.getIndex(itemIndexInInterval, nInterval))
                indexSampler += 1
            order[0][0] -= shift
            order.sort(reverse=True)
            if len(batch) == self.batchSize:
                yield batch
                batch = []

    def getIndex(self, x, iInterval):
        return x * self.sizeWindow + self.samplingIntervals[iInterval]

    def __len__(self):
        return sum(self.sizeSamplers) // self.batchSize
