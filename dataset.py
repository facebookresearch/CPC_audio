import os
import torch
from torch.utils.data import Dataset

import torchaudio


class AudioBatchData:

    # Work on this and on the sampler
    def __init__(self,
                 path,
                 seqNamesPath=None):
        """
        Args:
            path (string): path to the training dataset
            seqNamesPath (string): path to a file listing the sequences to load
        """

        self.dbPath = path

        seqNames = None
        if seqNamesPath is not None:
            seqNames = [p.replace('\n', '') + ".flac" for p in
                        open(seqNamesPath, 'r').readlines()]
        else:
            seqNames = self.findAllSeqs()

        self.loadAll(seqNames)

    def findAllSeqs(self):

        speakers = [f for f in os.listdir(self.dbPath)
                    if os.path.isdir(os.path.join(self.dbPath, f))]

        outSeqs = []
        for speaker in speakers:
            refPath = os.path.join(self.dbPath, speaker)
            chapters = os.listdir(refPath)
            for chapter in chapters:
                fullPath = os.path.join(refPath, chapter)
                outSeqs += [f for f in os.listdir(fullPath)
                            if os.path.splitext(f)[1] == '.flac']

        return outSeqs

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
        return len(self.data)

    def getNSpeakers(self):
        return len(self.speakers)

    def getSpeakerOffset(self, x):
        return self.speakerLabel[x]


class AudioBatchDataset(Dataset):

    def __init__(self,
                 batchData,
                 sizeWindow,
                 offset=0,
                 maxOffset=-1):

        self.batchData = batchData
        self.offset = offset
        self.sizeWindow = sizeWindow
        self.maxOffset = maxOffset

        if self.maxOffset <= 0:
            self.maxOffset = len(self.batchData)

        print(self.maxOffset)

    def __len__(self):

        return (self.maxOffset - self.offset) // self.sizeWindow

    def __getitem__(self, idx):

        windowOffset = self.offset + idx * self.sizeWindow
        speakerLabel = torch.tensor(
            self.batchData.getLabel(windowOffset), dtype=torch.long)

        return self.batchData.data[windowOffset:(self.sizeWindow
                                   + windowOffset)].view(1, -1), speakerLabel
