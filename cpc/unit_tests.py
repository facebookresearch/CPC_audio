# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import unittest
import torch
import os
import cpc.feature_loader as fl
from .dataset import AudioBatchData, findAllSeqs, filterSeqs
from nose.tools import eq_, ok_
from math import log


class TestDataLoader(unittest.TestCase):

    def setUp(self):

        self.seq_names = [(0, '6476/57446/6476-57446-0019.flac'),
                          (1, '5678/43303/5678-43303-0032.flac'),
                          (2, '1737/148989/1737-148989-0038.flac'),
                          (3, '6081/42010/6081-42010-0006.flac'),
                          (4, '1116/132851/1116-132851-0018.flac'),
                          (5, '5393/19218/5393-19218-0024.flac'),
                          (6, '4397/15668/4397-15668-0007.flac'),
                          (7, '696/92939/696-92939-0032.flac'),
                          (8, '3723/171115/3723-171115-0003.flac')]

        self.path_db = "/datasets01_101/LibriSpeech/022219/train-clean-100/"
        self.size_window = 20480
        self.speaker_list = list(set([x[0] for x in self.seq_names]))

    def testLoadData(self):

        test_data = AudioBatchData(self.path_db, self.size_window,
                                   self.seq_names, None, 9)
        assert(test_data.getNSpeakers() == 9)
        assert(test_data.getNSeqs() == 9)

    def testFindAllSeqs(self):
        seq_names, speakers = findAllSeqs(self.path_db,
                                          extension=".flac")
        expected_speakers = [f for f in os.listdir(self.path_db) if
                             os.path.isdir(os.path.join(self.path_db, f))]
        eq_(speakers, expected_speakers)

    def testFindAllSeqsCustomSpeakers(self):
        seq_names, speakers = findAllSeqs(self.path_db,
                                          extension=".flac",
                                          speaker_level=2)
        outDirs = [f for f in os.listdir(self.path_db) if
                   os.path.isdir(os.path.join(self.path_db, f))]
        expected_speakers = []
        for dir in outDirs:
            full_dir = os.path.join(self.path_db, dir)
            expected_speakers += [os.path.join(dir, f) for f in os.listdir(full_dir) if
                                  os.path.isdir(os.path.join(full_dir, f))]
        eq_(speakers, expected_speakers)

    def testFindAllSeqs0Speakers(self):
        seq_names, speakers = findAllSeqs("/datasets01_101/LibriSpeech/022219/train-clean-100/103/1240",
                                          extension=".flac")
        eq_(speakers, [''])

    def testFindAllSeqs0SpeakersForced(self):
        seq_names, speakers = findAllSeqs("/datasets01_101/LibriSpeech/022219/train-clean-100/",
                                          extension=".flac", speaker_level=0)
        eq_(speakers, [''])

    def testDataLoader(self):

        batch_size = 16
        path_seqs = "/datasets01_101/LibriSpeech/022219/LibriSpeech100_labels_split/test_split.txt"
        seq_names, speakers = findAllSeqs(self.path_db,
                                          extension=".flac")
        seq_names = filterSeqs(path_seqs, seq_names)

        test_data = AudioBatchData(self.path_db, self.size_window, seq_names,
                                   None, len(speakers))

        # Check the number of speakers
        nSpeaker = test_data.getNSpeakers()
        eq_(nSpeaker, 251)
        test_data_loader = test_data.getDataLoader(batch_size, "samespeaker",
                                                   True, numWorkers=2)
        for index, item in enumerate(test_data_loader):
            _, labels = item
            p = labels[0].item()
            eq_(torch.sum(labels == p), labels.size(0))

    def testPartialLoader(self):

        batch_size = 16
        test_data = AudioBatchData(self.path_db, self.size_window,
                                   self.seq_names, None, len(
                                       self.speaker_list),
                                   MAX_SIZE_LOADED=1000000)
        eq_(test_data.getNPacks(), 2)
        test_data_loader = test_data.getDataLoader(batch_size, "samespeaker",
                                                   True, numWorkers=2)
        visted_labels = set([])
        for index, item in enumerate(test_data_loader):
            _, labels = item
            p = labels[0].item()
            eq_(torch.sum(labels == p), labels.size(0))
            visted_labels.add(p)

        eq_(set(range(len(self.speaker_list))), visted_labels)


class TestPhonemParser(unittest.TestCase):

    def setUp(self):
        from .train import parseSeqLabels
        self.seqLoader = parseSeqLabels
        self.pathPhone = \
            "/private/home/mriviere/LibriSpeech/LibriSpeech100_labels_split/converted_aligned_phones.txt"

    def testSeqLoader(self):
        phone_data, nPhones = self.seqLoader(self.pathPhone)
        eq_(len(phone_data), 28539)
        eq_(phone_data['step'], 160)
        eq_(phone_data['4051-11218-0044'][43], 14)
        eq_(len(phone_data['4051-11218-0044']), 1119)
        eq_(nPhones, 41)

    def testSeqLabels(self):
        size_window = 640
        seq_names = [(0, '2911/12359/2911-12359-0007.flac'),
                     (1, '4051/11218/4051-11218-0044.flac')]
        speakers = list(set([x[0] for x in seq_names]))
        phone_data, _ = self.seqLoader(self.pathPhone)
        path_db = "/datasets01_101/LibriSpeech/022219/train-clean-100/"
        test_data = AudioBatchData(
            path_db, size_window, seq_names, phone_data, len(speakers))
        eq_(test_data.getPhonem(81280), [0, 0, 0, 0])
        eq_(test_data.getPhonem(84841), [0, 0, 0, 18])
        eq_(test_data.getPhonem(88201), [14, 14, 14, 14])


class TestLabelProcess(unittest.TestCase):

    def setUp(self):
        pass

    def testLabelCollapse(self):
        from .criterion.seq_alignment import collapseLabelChain

        input_chain = torch.tensor([[0, 0, 0, 1, 1, 2, 0, 2, 2],
                                    [1, 1, 1, 1, 1, 2, 2, 2, 0]],
                                   dtype=torch.int64)

        out_chain, sizes = collapseLabelChain(input_chain)
        target = torch.tensor([[0, 1, 2, 0, 2],
                               [1, 2, 0, 0, 0]],
                              dtype=torch.int64)
        target_size = torch.tensor([5, 3], dtype=torch.int64)

        eq_((out_chain - target).sum().item(), 0)
        eq_((target_size - sizes).sum().item(), 0)

    def test_beam_search(self):
        from .criterion.seq_alignment import beam_search
        import numpy as np
        blank_label = 2
        n_keep = 10

        data = np.array([[0.1, 0.2, 0.],
                         [0.4, 0.2, 0.6],
                         [0.01, 0.3, 0.]])

        output = beam_search(data, n_keep, blank_label)

        expected_pos_output = [(0.036, [1, 1]), (0.0004, [0]), (0.012, [1]),
                               (0.024, [1, 0, 1]), (0.0002, [
                                   0, 1, 0]), (0.0, [1, 1, 1]),
                               (0.0, [1, 1, 0]), (0.0006,
                                                  [0, 0]), (0.036, [0, 1]),
                               (0.0024, [1, 0])]
        expected_pos_output.sort(reverse=True)

        for index, item in enumerate(expected_pos_output):
            eq_(item[1], output[index][1])
            ok_(abs(item[0] - output[index][0]) < 1e-08)

    def test_big_beam_search(self):
        from .criterion.seq_alignment import beam_search
        import numpy as np
        blank_label = 11
        n_keep = 10

        data = np.array([[0.1, 0.2,  0.,  0.,  0.,  0.,  0., 0.01,  0., 0.1, 0.99, 0.1],
                         [0.1, 0.2, 0.6, 0.1, 0.9,  0.,  0., 0.01,  0., 0.9, 1., 0.]])

        output = beam_search(data, n_keep, blank_label)[0]

        expected_output = (1.09, [10])
        eq_(output[0], expected_output[0])
        eq_(output[1], expected_output[1])


class TestPER(unittest.TestCase):

    def setUp(self):
        pass

    def testPER(self):
        from .criterion.seq_alignment import get_seq_PER

        ref_seq = [0, 1, 1, 2, 0, 2, 2]
        pred_seq = [1, 1, 2, 2, 0, 0]

        expected_PER = 4. / 7.
        eq_(get_seq_PER(ref_seq, pred_seq), expected_PER)


class TestEncoderBuilder(unittest.TestCase):

    def setUp(self):
        from cpc.cpc_default_config import get_default_cpc_config
        self.default_args = get_default_cpc_config()

    def testBuildMFCCEncoder(self):
        from cpc.model import MFCCEncoder
        self.default_args.encoder_type = 'mfcc'
        self.default_args.hiddenEncoder = 30

        test_encoder = fl.getEncoder(self.default_args)
        ok_(isinstance(test_encoder, MFCCEncoder))
        eq_(test_encoder.dimEncoded, 30)

    def testBuildLFBEnconder(self):
        from cpc.model import LFBEnconder
        self.default_args.encoder_type = 'lfb'
        self.default_args.hiddenEncoder = 12

        test_encoder = fl.getEncoder(self.default_args)
        ok_(isinstance(test_encoder, LFBEnconder))
        eq_(test_encoder.dimEncoded, 12)

    def testBuildCPCEncoder(self):
        from cpc.model import CPCEncoder
        test_encoder = fl.getEncoder(self.default_args)
        ok_(isinstance(test_encoder, CPCEncoder))
        eq_(test_encoder.dimEncoded, 256)


class TestARBuilder(unittest.TestCase):

    def setUp(self):
        from cpc.cpc_default_config import get_default_cpc_config
        self.default_args = get_default_cpc_config()

    def testbuildBertAR(self):
        from cpc.model import BiDIRARTangled
        self.default_args.cpc_mode = 'bert'

        test_ar = fl.getAR(self.default_args)
        ok_(isinstance(test_ar, BiDIRARTangled))

    def testbuildNoAR(self):
        from cpc.model import NoAr
        self.default_args.arMode = 'no_ar'

        test_ar = fl.getAR(self.default_args)
        ok_(isinstance(test_ar, NoAr))

    def testbuildNoAR(self):
        from cpc.model import CPCAR
        self.default_args.arMode = 'LSTM'
        test_ar = fl.getAR(self.default_args)
        ok_(isinstance(test_ar, CPCAR))
        ok_(isinstance(test_ar.baseNet, torch.nn.LSTM))

    def testbuildNoAR(self):
        from cpc.model import CPCAR
        self.default_args.arMode = 'GRU'
        test_ar = fl.getAR(self.default_args)
        ok_(isinstance(test_ar, CPCAR))
        ok_(isinstance(test_ar.baseNet, torch.nn.GRU))

    def testbuildNoAR(self):
        from cpc.model import CPCAR
        self.default_args.arMode = 'RNN'
        test_ar = fl.getAR(self.default_args)
        ok_(isinstance(test_ar, CPCAR))
        ok_(isinstance(test_ar.baseNet, torch.nn.RNN))
