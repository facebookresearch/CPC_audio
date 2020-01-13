import unittest
import torch
import os
from .dataset import AudioBatchData, findAllSeqs, filterSeqs
from nose.tools import eq_, ok_
from math import log
# from phone_cluster_correlation import convertToProbaMatrix, getMutualInfo, \
#                                       getEntropy, featureLabelToRepMat, \
#                                       getSegsStats


class TestDataLoader(unittest.TestCase):

    def setUp(self):

        self.seqNames = [(0, '6476/57446/6476-57446-0019.flac'),
                         (1, '5678/43303/5678-43303-0032.flac'),
                         (2, '1737/148989/1737-148989-0038.flac'),
                         (3, '6081/42010/6081-42010-0006.flac'),
                         (4, '1116/132851/1116-132851-0018.flac'),
                         (5, '5393/19218/5393-19218-0024.flac'),
                         (6, '4397/15668/4397-15668-0007.flac'),
                         (7, '696/92939/696-92939-0032.flac'),
                         (8, '3723/171115/3723-171115-0003.flac')]

        self.pathDB = "/datasets01_101/LibriSpeech/022219/train-clean-100/"
        self.sizeWindow = 20480
        self.speakerList = list(set([x[0] for x in self.seqNames]))

    def testLoadData(self):

        testData = AudioBatchData(self.pathDB, self.sizeWindow,
                                  self.seqNames, None, 9)
        assert(testData.getNSpeakers() == 9)
        assert(testData.getNSeqs() == 9)

    def testFindAllSeqs(self):
        seqNames, speakers = findAllSeqs(self.pathDB,
                                         extension=".flac")
        expectedSpeakers = [f for f in os.listdir(self.pathDB) if
                            os.path.isdir(os.path.join(self.pathDB, f))]
        eq_(speakers, expectedSpeakers)

    def testFindAllSeqsCustomSpeakers(self):
        seqNames, speakers = findAllSeqs(self.pathDB,
                                         extension=".flac",
                                         speaker_level=2)
        outDirs = [f for f in os.listdir(self.pathDB) if
                            os.path.isdir(os.path.join(self.pathDB, f))]
        expectedSpeakers = []
        for dir in outDirs:
            full_dir = os.path.join(self.pathDB, dir)
            expectedSpeakers += [os.path.join(dir, f) for f in os.listdir(full_dir) if
                                os.path.isdir(os.path.join(full_dir, f))]
        eq_(speakers, expectedSpeakers)

    def testFindAllSeqs0Speakers(self):
        seqNames, speakers = findAllSeqs("/datasets01_101/LibriSpeech/022219/train-clean-100/103/1240",
                                         extension=".flac")
        eq_(speakers, [''])

    def testFindAllSeqs0SpeakersForced(self):
        seqNames, speakers = findAllSeqs("/datasets01_101/LibriSpeech/022219/train-clean-100/",
                                         extension=".flac", speaker_level=0)
        eq_(speakers, [''])


    def testDataLoader(self):

        batchSize = 16
        pathSeqs = "/datasets01_101/LibriSpeech/022219/LibriSpeech100_labels_split/test_split.txt"
        seqNames, speakers = findAllSeqs(self.pathDB,
                                         extension=".flac")
        seqNames = filterSeqs(pathSeqs, seqNames)

        testData = AudioBatchData(self.pathDB, self.sizeWindow, seqNames,
                                  None, len(speakers))

        # Check the number of speakers
        nSpeaker = testData.getNSpeakers()
        eq_(nSpeaker, 251)
        testDataLoader = testData.getDataLoader(batchSize, "samespeaker",
                                                True, numWorkers=2)
        for index, item in enumerate(testDataLoader):
            _, labels = item
            p = labels[0].item()
            eq_(torch.sum(labels == p), labels.size(0))

    def testPartialLoader(self):

        batchSize = 16
        testData = AudioBatchData(self.pathDB, self.sizeWindow,
                                  self.seqNames, None, len(self.speakerList),
                                  MAX_SIZE_LOADED=1000000)
        eq_(testData.getNPacks(), 2)
        testDataLoader = testData.getDataLoader(batchSize, "samespeaker",
                                                True, numWorkers=2)
        vistedLabels = set([])
        for index, item in enumerate(testDataLoader):
            _, labels = item
            p = labels[0].item()
            eq_(torch.sum(labels == p), labels.size(0))
            vistedLabels.add(p)

        eq_(set(range(len(self.speakerList))), vistedLabels)


class TestPhonemParser(unittest.TestCase):

    def setUp(self):
        from .train import parseSeqLabels
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
        speakers = list(set([x[0] for x in seqNames]))
        phoneData, _ = self.seqLoader(self.pathPhone)
        pathDB = "/datasets01_101/LibriSpeech/022219/train-clean-100/"
        testData = AudioBatchData(
            pathDB, sizeWindow, seqNames, phoneData, speakers)
        eq_(testData.getPhonem(81280), [0, 0, 0, 0])
        eq_(testData.getPhonem(84841), [0, 0, 0, 18])
        eq_(testData.getPhonem(88201), [14, 14, 14, 14])


class TestLabelProcess(unittest.TestCase):

    def setUp(self):
        pass

    def testLabelCollapse(self):
        from .criterion.seq_alignment import collapseLabelChain

        inputChain = torch.tensor([[0, 0, 0, 1, 1, 2, 0, 2, 2],
                                   [1, 1, 1, 1, 1, 2, 2, 2, 0]],
                                  dtype=torch.int64)

        outChain, sizes = collapseLabelChain(inputChain)
        target = torch.tensor([[0, 1, 2, 0, 2],
                               [1, 2, 0, 0, 0]],
                               dtype=torch.int64)
        targetSize = torch.tensor([5,3], dtype=torch.int64)

        eq_((outChain - target).sum().item(), 0)
        eq_((targetSize - sizes).sum().item(), 0)

    def test_beam_search(self):
        from .criterion.seq_alignment import beam_search
        import numpy as np
        blankLabel = 2
        nKeep = 10

        data = np.array([[0.1, 0.2, 0.],
                         [0.4, 0.2, 0.6],
                         [0.01, 0.3, 0.]])

        output = beam_search(data, nKeep, blankLabel)

        expectedPosOutput = [(0.036, [1, 1]), (0.0004, [0]), (0.012, [1]),
                             (0.024, [1, 0, 1]), (0.0002, [0,1,0]), (0.0, [1, 1, 1]),
                             (0.0, [1,1,0]), (0.0006, [0,0]), (0.036,[0,1]),
                             (0.0024, [1,0])]
        expectedPosOutput.sort(reverse=True)

        for index, item in enumerate(expectedPosOutput):
            eq_(item[1], output[index][1])
            ok_(abs(item[0] - output[index][0])< 1e-08)

    def test_big_beam_search(self):
        from .criterion.seq_alignment import beam_search
        import numpy as np
        blankLabel = 11
        nKeep = 10

        data = np.array([[0.1, 0.2,  0.,  0.,  0.,  0.,  0., 0.01,  0., 0.1, 0.99, 0.1],
                         [0.1, 0.2, 0.6, 0.1, 0.9,  0.,  0., 0.01,  0., 0.9, 1., 0.]])

        output = beam_search(data, nKeep, blankLabel)[0]

        expectedOutput = (1.09, [10])
        eq_(output[0], expectedOutput[0])
        eq_(output[1], expectedOutput[1])


class TestPER(unittest.TestCase):

    def setUp(self):
        pass

    def testPER(self):
        from .criterion.seq_alignment import get_seq_PER

        ref_seq = [0, 1, 1, 2, 0, 2, 2]
        pred_seq = [1, 1, 2, 2, 0, 0]

        expected_PER = 4. / 7.
        eq_(get_seq_PER(ref_seq, pred_seq), expected_PER)
