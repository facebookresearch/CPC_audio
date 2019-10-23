import unittest
import torch
from dataset import AudioBatchData, findAllSeqs, filterSeqs
from nose.tools import eq_, ok_
from math import log
from dim_reduction import PCA
from phone_cluster_correlation import convertToProbaMatrix, getMutualInfo, \
                                      getEntropy, featureLabelToRepMat, \
                                      getSegsStats


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

        self.pathDB = "/datasets01_101/LibriSpeech/022219/train-clean-100/"
        self.sizeWindow = 20480
        self.speakerList = list(set([x[0] for x in self.seqNames]))

    def testLoadData(self):

        testData = AudioBatchData(self.pathDB, self.sizeWindow,
                                  self.seqNames, None, self.speakerList)
        assert(testData.getNSpeakers() == 9)
        assert(testData.getNSeqs() == 9)

    def testDataLoader(self):

        batchSize = 16
        pathSeqs = "/datasets01_101/LibriSpeech/022219/LibriSpeech100_labels_split/test_split.txt"
        seqNames, speakers = findAllSeqs(self.pathDB, recursionLevel=2,
                                         extension=".flac")
        seqNames = filterSeqs(pathSeqs, seqNames)

        testData = AudioBatchData(self.pathDB, self.sizeWindow, seqNames,
                                  None, list(speakers))

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
                                  self.seqNames, None, self.speakerList,
                                  MAX_SIZE_LOADED=1000000,
                                  GROUP_SIZE_LOADED=2)
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
        speakers = list(set([x[0] for x in seqNames]))
        phoneData, _ = self.seqLoader(self.pathPhone)
        pathDB = "/datasets01_101/LibriSpeech/022219/train-clean-100/"
        testData = AudioBatchData(
            pathDB, sizeWindow, seqNames, phoneData, speakers)
        eq_(testData.getPhonem(81280), [0, 0, 0, 0])
        eq_(testData.getPhonem(84841), [0, 0, 0, 18])
        eq_(testData.getPhonem(88201), [14, 14, 14, 14])


class TestPCA(unittest.TestCase):

    def setUp(self):
        self.N = 6
        self.k = 3

    def testPCA(self):

        module = PCA(self.k, normalize=False)
        data = torch.tensor([[ 0.1681, -0.6360, -0.5347],
                             [-1.7113,  0.5962, -0.5708],
                             [-0.4865,  0.7551, -0.0701],
                             [ 0.4084, -0.2050,  0.5220],
                             [-2.4044, -1.5335, -0.2291],
                             [ 0.3060,  0.4486, -1.7393]])

        module.update(data[:3])
        module.update(data[3:])

        eq_(module.N, 6)
        ok_((module.mean - torch.tensor([-3.7197, -0.5747, -2.6220])).norm() < 1e-3)
        ok_((module.var - torch.tensor([[ 9.2351,  2.2462,  1.1529],
                                        [ 2.2462,  3.9251, -0.5890],
                                        [ 1.1529, -0.5890,  3.9669]])).norm() < 1e-3)

        module.build()
        ref_mean = torch.tensor([-0.6199, -0.0958, -0.4370])

        ok_((module.mean - ref_mean).norm() < 1e-3)
        ok_((module.var - torch.tensor([[ 1.1548,  0.3150, -0.0788],
                                        [ 0.3150,  0.6450, -0.1400],
                                        [-0.0788, -0.1400,  0.4702]])).norm() < 1e-3)

        testItem = torch.tensor([[ 1., 0., 0.], [0.,  0., 1.]]) + ref_mean

        projectedItems = module(testItem)
        expectedProjection = torch.tensor([ [ 0.1665,  0.4362, -0.8843], [-0.7785,  0.6085,  0.1536]])
        ok_((projectedItems - expectedProjection).norm() < 1e-3)

class TestKMean(unittest.TestCase):

    def setUp(self):
        pass

    def testKMeanStep(self):
        from clustering import kMeanCluster
        Ck = torch.tensor([[1, 0, 2],
                           [-1, 3, 1]]).float()

        cluster = kMeanCluster(Ck.view(1, 2, 3))
        a = torch.tensor([[0,0,1], [0,0,0], [1,1,1]]).float()
        a = a.view(3, 1, 3)
        norm = cluster(a).view(3, 2)

        eq_(norm[0,0], 2. / 3.)
        eq_(norm[0,1], 10. / 3.)
        eq_(norm[1,0], 5. / 3.)
        eq_(norm[1,1], 11. / 3.)
        eq_(norm[2,0], 2. / 3.)
        eq_(norm[2,1], 8. / 3.)

    def testKMeanLoss(self):
        from criterion import ClusteringSoftMax
        test = ClusteringSoftMax(2,3, 0, 10)
        test.clusters.Ck.copy_(torch.tensor([[1, 0, 2],
                               [-1, 3, 1]]).float())

        a = torch.tensor([[0,0,1]]).float()
        loss = test(a.view(1, 1, 3))
        ok_(abs(loss.sum().item() - 0.06717577604314462) < 1e-4)

class TestDeepEmbedded(unittest.TestCase):

    def setUp(self):
        pass


class TestStats(unittest.TestCase):

    def setUp(self):
        pass

    def testProba(self):

        dimPhone = 4
        dimCluster = 3
        refMatData = torch.tensor([[1, 0, 22], [2,4,5], [31, 3, 12], [0,0, 5]], dtype= torch.long)
        probaMat = convertToProbaMatrix(refMatData, 'full')

        eq_(probaMat[0, 0], 1. / 85.)
        eq_(probaMat[0, 1], 0.)
        eq_(probaMat[0, 2], 22. / 85.)

        eq_(probaMat[1, 0], 2. / 85.)
        eq_(probaMat[1, 1], 4. / 85.)
        eq_(probaMat[1, 2], 5. / 85.)

        eq_(probaMat[2, 0], 31. / 85.)
        eq_(probaMat[2, 1], 3. / 85.)
        eq_(probaMat[2, 2], 12. / 85.)

        eq_(probaMat[3, 0], 0.)
        eq_(probaMat[3, 1], 0.)
        eq_(probaMat[3, 2], 5. / 85.)

        pPhone = probaMat.sum(dim=1)
        pCluster = probaMat.sum(dim=0)

        eq_(pPhone[0], 23. / 85.)
        eq_(pPhone[1], 11. / 85.)
        eq_(pPhone[2], 46. / 85.)
        eq_(pPhone[3], 5. / 85.)

        eq_(pCluster[0], 34. / 85.)
        eq_(pCluster[1], 7. / 85.)
        eq_(pCluster[2], 44. / 85.)

        entropyPhone = - (pPhone[0] * log(pPhone[0]) \
                          + pPhone[1] * log(pPhone[1]) \
                          + pPhone[2] * log(pPhone[2]) \
                          + pPhone[3] * log(pPhone[3]))

        eq_(getEntropy(pPhone), entropyPhone)

        iPH =   probaMat[0, 0] * log(probaMat[0, 0] / (pPhone[0] * pCluster[0])) \
              + probaMat[0, 2] * log(probaMat[0, 2] / (pPhone[0] * pCluster[2])) \
              \
              + probaMat[1, 0] * log(probaMat[1, 0] / (pPhone[1] * pCluster[0])) \
              + probaMat[1, 1] * log(probaMat[1, 1] / (pPhone[1] * pCluster[1])) \
              + probaMat[1, 2] * log(probaMat[1, 2] / (pPhone[1] * pCluster[2])) \
              \
              + probaMat[2, 0] * log(probaMat[2, 0] / (pPhone[2] * pCluster[0])) \
              + probaMat[2, 1] * log(probaMat[2, 1] / (pPhone[2] * pCluster[1])) \
              + probaMat[2, 2] * log(probaMat[2, 2] / (pPhone[2] * pCluster[2])) \
              \
              + probaMat[3, 2] * log(probaMat[3, 2] / (pPhone[3] * pCluster[2]))

        ok_(abs(iPH.item() - getMutualInfo(probaMat, pPhone.view(dimPhone, 1),
            pCluster.view(1, dimCluster)).item()) < 1e-05)

        # Conditionnal probabilities

        phoneIfCluster = convertToProbaMatrix(refMatData, 'p_phoneIfcluster')
        refMat = [ [ 1. / 34. ,      0., 22. / 44.],
                   [ 2. / 34.,  4. / 7.,  5. / 44.],
                   [31. / 34.,  3. / 7., 12. / 44.],
                   [       0.,       0.,  5. / 44.]]

        for p in range(dimPhone):
           for c in range(dimCluster):
               eq_(phoneIfCluster[p, c], refMat[p][c])

        refMat = [ [ 1. / 23. ,       0., 22. / 23.],
                   [ 2. / 11.,  4. / 11.,  5. / 11.],
                   [31. / 46.,  3. / 46., 12. / 46.],
                   [       0.,       0.,         1.]]


        clusterIfPhone = convertToProbaMatrix(refMatData, 'p_clusterIfphone')
        for p in range(dimPhone):
           for c in range(dimCluster):
               eq_(clusterIfPhone[p, c], refMat[p][c])


    def testRepMatMaker(self):

        dimPhone = 4
        dimCluster = 3
        BatchSIZE = 2
        SeqSize = 5

        features = torch.tensor([[[0, 1, 0],
                                  [0, 0, 1],
                                  [0, 1, 0],
                                  [1, 0, 0],
                                  [1, 0, 0]],

                                 [[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]]], dtype=torch.long)

        labels = torch.tensor([[0, 0, 3, 2, 1],
                               [1, 3, 3, 0, 0]], dtype=torch.long)

        print(labels.size(), features.size())
        repMat = featureLabelToRepMat(features, labels, dimPhone, dimCluster)


        expectedOutput = [[1, 1, 2],
                          [2, 0, 0],
                          [1, 0, 0],
                          [0, 3, 0]]

        for p in range(dimPhone):
           for c in range(dimCluster):
               eq_(repMat[p,c], expectedOutput[p][c])

    def testSegStats(self):

        inputSeq = torch.tensor([[0, 1, 0],
                                 [0, 1, 0],
                                 [1, 0, 0],
                                 [1, 0, 0],
                                 [1, 0, 0]], dtype = torch.long)

        segSize, segFreq = getSegsStats(inputSeq)
        eq_(segSize[0], 3)
        eq_(segSize[1], 2)
        eq_(segSize[2], 0)

        eq_(segFreq[0], 1)
        eq_(segFreq[1], 1)
        eq_(segFreq[2], 0)


class TestLabelProcess(unittest.TestCase):

    def setUp(self):
        pass

    def testLabelCollapse(self):
        from criterion import collapseLabelChain

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

    def testClearBlank(self):
        from criterion import removeBlanks

        blankLabel = 22
        inputChain = torch.tensor([0, 0, 22, 22, 1, 22, 0, 0, 0],
                                  dtype=torch.int64)
        inputSize = 6
        outChain, outSize = removeBlanks(inputChain, inputSize, blankLabel)

        targetChain = torch.tensor([0, 0, 1, 0, 0, 0],
                                   dtype=torch.int64)

        eq_(outSize.item(), 3)
        eq_((outChain - targetChain).sum(), 0)

    def testBeamSearch(self):
        from beam_search import beamSearch
        import numpy as np
        blankLabel = 2
        nKeep = 10

        data = np.array([[0.1, 0.2, 0.],
                         [0.4, 0.2, 0.6],
                         [0.01, 0.3, 0.]])

        output = beamSearch(data, nKeep, blankLabel)

        expectedPosOutput = [(0.036, [1, 1]), (0.0004, [0]), (0.012, [1]),
                             (0.024, [1, 0, 1]), (0.0002, [0,1,0]), (0.0, [1, 1, 1]),
                             (0.0, [1,1,0]), (0.0006, [0,0]), (0.036,[0,1]),
                             (0.0024, [1,0])]
        expectedPosOutput.sort(reverse=True)

        for index, item in enumerate(expectedPosOutput):
            eq_(item[1], output[index][1])
            ok_(abs(item[0] - output[index][0])< 1e-08)

    def testBigBeamSearch(self):
        from beam_search import beamSearch
        import numpy as np
        blankLabel = 11
        nKeep = 10

        data = np.array([[0.1, 0.2,  0.,  0.,  0.,  0.,  0., 0.01,  0., 0.1, 0.99, 0.1],
                         [0.1, 0.2, 0.6, 0.1, 0.9,  0.,  0., 0.01,  0., 0.9, 1., 0.]])

        output = beamSearch(data, nKeep, blankLabel)[0]

        expectedOutput = (1.09, [10])
        eq_(output[0], expectedOutput[0])
        eq_(output[1], expectedOutput[1])


class TestPER(unittest.TestCase):

    def setUp(self):
        pass

    def testPER(self):
        from eval_PER import getSeqPER

        refSeq = [0, 1, 1, 2, 0, 2, 2]
        predSeq = [1, 1, 2, 2, 0, 0]

        expectedPER = 4. / 7.
        eq_(getSeqPER(refSeq, predSeq), expectedPER)
