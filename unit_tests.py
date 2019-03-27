import unittest
import torch
from train import findAllSeqs, filterSeqs
from dataset import SameSpeakerSampler, AudioBatchData
from nose.tools import eq_


class TestDataLoader(unittest.TestCase):

    def setUp(self):

        self.seqNames = [(6476, '6476/57446/6476-57446-0019.flac'),
                         (5678, '5678/43303/5678-43303-0032.flac'),
                         (1737, '1737/148989/1737-148989-0038.flac'),
                         (6081, '6081/42010/6081-42010-0006.flac'),
                         (1116, '1116/132851/1116-132851-0018.flac'),
                         (5393, '5393/19218/5393-19218-0024.flac'),
                         (4397, '4397/15668/4397-15668-0007.flac'),
                         (696, '696/92939/696-92939-0032.flac'),
                         (3723, '3723/171115/3723-171115-0003.flac')]

        self.pathDB = "/datasets01/LibriSpeech/022219/train-clean-100/"
        self.sizeWindow = 20480

    def testLoadData(self):

        testData = AudioBatchData(self.pathDB, self.sizeWindow,
                                  self.seqNames, None)
        assert(testData.getNSpeakers() == 9)
        assert(testData.getNSeqs() == 9)

    def testDataLoader(self):

        batchSize = 16
        groupSize = 4
        pathSeqs = "/datasets01/LibriSpeech/022219/LibriSpeech100_labels_split/test_split.txt"
        seqNames = findAllSeqs(self.pathDB, recursionLevel=2, extension=".flac")
        seqNames = filterSeqs(pathSeqs, seqNames)

        testData = AudioBatchData(self.pathDB, self.sizeWindow, seqNames, None)

        # Check the number of speakers
        nSpeaker = testData.getNSpeakers()
        eq_(nSpeaker, 251)
        testSampler = testData.getSampler(
            batchSize, "samespeaker", True)
        testDataLoader = torch.utils.data.DataLoader(testData,
                                                     batch_sampler=testSampler,
                                                     num_workers=2)

        for index, item in enumerate(testDataLoader):

            _, labels = item
            p = labels[0].item()
            print(labels)
            eq_(torch.sum(labels == p), labels.size(0))


class TestPhonemParser(unittest.TestCase):

    def setUp(self):
        from train import parseSeqLabels
        self.seqLoader = parseSeqLabels
        self.pathPhone = \
            "/private/home/mriviere/LibriSpeech/LibriSpeech100_labels_split/converted_aligned_phones.txt"

    def testSeqLoader(self):
        phoneData, nPhones = self.seqLoader(self.pathPhone)
        eq_(len(phoneData), 28539)
        eq_(phoneData['step'], 160)
        eq_(phoneData['4051-11218-0044'][43], 14)
        eq_(len(phoneData['4051-11218-0044']), 1119)
        eq_(nPhones, 41)

    def testSeqLabels(self):
        sizeWindow = 640
        seqNames = [(4051, '4051/11218/4051-11218-0044.flac'),
                    (2911, '2911/12359/2911-12359-0007.flac')]
        phoneData, _ = self.seqLoader(self.pathPhone)
        pathDB = "/datasets01/LibriSpeech/022219/train-clean-100/"
        testData = AudioBatchData(pathDB, sizeWindow, seqNames, phoneData)
        eq_(testData.getPhonem(81280), [0, 0, 0, 0])
        eq_(testData.getPhonem(84841), [0, 0, 0, 18])
        eq_(testData.getPhonem(88201), [14, 14, 14, 14])
