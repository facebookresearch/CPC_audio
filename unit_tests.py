import unittest
import torch
from dataset import AudioBatchSampler, AudioBatchData


class TestSampler(unittest.TestCase):
    def test_api(self):
        batchSize = 16
        groupSize = 4
        samplingIntervals = [0, 256, 459, 687, 1908, 2230]
        sizeWindow = 4

        testSampler = AudioBatchSampler(batchSize,
                                        groupSize,
                                        samplingIntervals,
                                        sizeWindow,
                                        False)
        shift = 0
        indexes = []
        for item in testSampler:
            assert(q not in indexes for q in item)
            indexes += item
            shift += 1

        assert(shift == len(testSampler))


class TestDataLoader(unittest.TestCase):

    def setUp(self):

        self.seqNames = ['6476-57446-0019.flac', '5678-43303-0032.flac',
                         '1737-148989-0038.flac', '6081-42010-0006.flac',
                         '1116-132851-0018.flac', '5393-19218-0024.flac',
                         '4397-15668-0007.flac', '696-92939-0032.flac',
                         '3723-171115-0003.flac']

        self.pathDB = "/datasets01/LibriSpeech/022219/train-clean-100/"
        self.sizeWindow = 20480

    def test_loadData(self):

        testData = AudioBatchData(self.pathDB, self.sizeWindow, self.seqNames)
        assert(testData.getNSpeakers() == 9)
        assert(testData.getNSeqs() == 9)

    def test_sampleSpeaker(self):

        batchSize = 16
        groupSize = 4
        pathSeqs = "/datasets01/LibriSpeech/022219/LibriSpeech100_labels_split/test_split.txt"
        seqNames = [p.replace('\n', '') + ".flac" for p in
                    open(pathSeqs, 'r').readlines()]

        testData = AudioBatchData(self.pathDB, self.sizeWindow, seqNames)

        # Check the number of speakers
        nSpeaker = testData.getNSpeakers()

        nValidBatch = 0
        nItemLabels = [0 for x in range(nSpeaker)]
        testSampler = testData.getSampler(batchSize, groupSize, "speaker", True)
        testDataLoader = torch.utils.data.DataLoader(testData,
                                                     batch_sampler=testSampler,
                                                     num_workers=2)

        for index, item in enumerate(testDataLoader):

            data, labels = item
            isValid = True
            for i in range(batchSize):
                p = labels[i].item()
                isValid = isValid and torch.sum(labels == p) >= groupSize \
                    and torch.sum(labels != p) > 0
                nItemLabels[p] += 1
            if isValid:
                nValidBatch += 1

        # Since the speakers are not evenly represented, we can't reach
        # 100% validity
        assert(nValidBatch / len(testSampler) > 0.9)

        # Assert that there isn't enough data remaining to make a batch
        # + label coherence
        r = 0
        for index, value in enumerate(nItemLabels):
            maxSpeakerValue = testSampler.getSpeakerMaxSize(index)
            remaining = maxSpeakerValue - value
            assert(remaining < batchSize and remaining >= 0)
            r += remaining

        assert(r < batchSize)
