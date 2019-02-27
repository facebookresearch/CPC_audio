import unittest
import torch
from dataset import AudioBatchSampler, AudioBatchData


class TestSampler(unittest.TestCase):
    def coin(self):
        batchSize = 16
        groupSize = 4
        samplingIntervals = [0, 256, 459, 687, 1908, 2230]
        sizeWindow = 4

        testSampler = AudioBatchSampler(batchSize,
                                        groupSize,
                                        samplingIntervals,
                                        sizeWindow)
        shift = 0
        indexes = []
        for item in testSampler:
            assert(q not in indexes for q in item)
            indexes += item
            shift += 1

        assert(shift == len(testSampler))

class TestDataLoader(unittest.TestCase):

    def test_api(self):

        def base(pathSeqs):
            batchSize = 16
            groupSize = 4
            sizeWindow = 20480
            pathDB = "/datasets01/LibriSpeech/022219/train-clean-100/"
            seqNames = [p.replace('\n', '') + ".flac" for p in
                        open(pathSeqs, 'r').readlines()]

            testData = AudioBatchData(pathDB, sizeWindow, seqNames)
            testSampler = testData.getSampler(batchSize, groupSize)

            # Check the number of speakers
            nSpeaker = testData.getNSpeakers()
            assert(nSpeaker == 251)

            testDataLoader = torch.utils.data.DataLoader(testData,
                                                         batch_sampler=testSampler,
                                                         num_workers=2)
            nValidBatch = 0
            nItemLabels = [0 for x in range(nSpeaker)]
            pp = 0

            for index, item in enumerate(testDataLoader):

                data, labels = item
                isValid = True
                for i in range(batchSize):
                    p = labels[i].item()
                    isValid = isValid and torch.sum(labels == p) >= groupSize \
                                and torch.sum(labels != p) > 0
                    nItemLabels[p] += 1
                if isValid:
                    nValidBatch+= 1

                pp+=1

            # Since the speakers are not evenly represented, we can't reach 100%
            # validity
            print(pp, len(testSampler), len(testDataLoader))
            assert(nValidBatch / len(testSampler) > 0.9)

            # Assert that there isn't enough data remaining to make a batch
            # + label coherence
            r = 0
            for index, value in enumerate(nItemLabels):
                maxSpeakerValue = testData.getSpeakerMaxSize(index)
                remaining = maxSpeakerValue - value
                assert(remaining < batchSize and remaining >=0)
                r+= remaining

            assert(r < batchSize)

        base("/private/home/mriviere/LibriSpeech/LibriSpeech100_labels_split/test_split.txt")
        base("/private/home/mriviere/LibriSpeech/LibriSpeech100_labels_split/train_split.txt")
