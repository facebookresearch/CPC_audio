import os
import torch
from torch.utils.data import Dataset

import torchaudio
import math

class AudioBatchData:

    # Work on this and on the sampler
    def __init__(self,
                 path):

        self.dbPath = path
        self.loadAll()

    def loadAll(self):

        # Speakers
        self.speakers = [f for f in os.listdir(self.dbPath) \
                         if os.path.isdir(os.path.join(self.dbPath, f))]

        self.speakers =self.speakers[:5]

        # Labels
        self.speakerLabel = [0]
        self.seqLabel = [0]

        # Data
        self.data = []

        itemIndex = 0
        seqIndex = 0
        speakerIndex=0

        for indexSpeaker, speaker in enumerate(self.speakers):
            refPath = os.path.join(self.dbPath, speaker)
            chapters = [ f for f in os.listdir(refPath) \
                        if os.path.isdir(os.path.join(refPath, f))]

            for chapter in chapters:
                chapterPath = os.path.join(refPath, chapter)
                #Debugging only

                for seqName in os.listdir(chapterPath):
                    if os.path.splitext(seqName)[1] != '.flac':
                        continue

                    seqPath = os.path.join(chapterPath, seqName)
                    seq = torchaudio.load(seqPath)[0].view(-1)

                    sizeSeq = seq.size(0)
                    seqIndex+= sizeSeq
                    speakerIndex+= sizeSeq

                    self.data.append(seq)
                    self.seqLabel.append(seqIndex)
                    itemIndex+=1

            self.speakerLabel.append(speakerIndex)

        self.data = torch.cat(self.data, dim = 0)

    def getLabel(self, idx):

       idSpeaker = next(x[0] for x in enumerate(self.speakerLabel) if x[1] > idx) -1
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
                 offset =0,
                 maxOffset = -1):

        self.batchData = batchData
        self.offset = offset
        self.sizeWindow = sizeWindow
        self.maxOffset = maxOffset

        if self.maxOffset <= 0:
            self.maxOffset = len(self.batchData)

        print(self.maxOffset)

    def __len__(self):

        return int(math.floor((self.maxOffset - self.offset) / self.sizeWindow))

    def __getitem__(self, idx):

        windowOffset = self.offset + idx * self.sizeWindow
        speakerLabel = torch.tensor(self.batchData.getLabel(windowOffset), dtype=torch.long)

        return self.batchData.data[windowOffset:(self.sizeWindow + windowOffset)].view(1, -1), speakerLabel

# On doit pouvoir sampler
# par fichier audio, par speaker, et uniform
# faire les UTs

# Pipeline
# pre-training puis downstream task
