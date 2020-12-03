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
from pathlib import Path


class TestDataLoader(unittest.TestCase):

    def setUp(self):

        self.seq_names = ['6476/57446/6476-57446-0019.flac',
                          '5678/43303/5678-43303-0032.flac',
                          '5678/43303/5678-43303-0024.flac',
                          '5678/43301/5678-43301-0021.flac',
                          '5393/19218/5393-19218-0024.flac',
                          '4397/15668/4397-15668-0007.flac',
                          '4397/15668/4397-15668-0003.flac']

        self.test_data_dir = Path(__file__).parent / 'test_data'
        self.path_db = self.test_data_dir / 'test_db'
        self.seq_list = self.test_data_dir / 'seq_list.txt'
        self.size_window = 20480

    def testFindAllSeqs(self):
        seq_names, speakers = findAllSeqs(str(self.path_db),
                                          extension=".flac")
        expected_output = [(0, '2911/12359/2911-12359-0007.flac'),
                           (1, '4051/11218/4051-11218-0044.flac'),
                           (2, '4397/15668/4397-15668-0003.flac'),
                           (2, '4397/15668/4397-15668-0007.flac'),
                           (3, '5393/19218/5393-19218-0024.flac'),
                           (4, '5678/43301/5678-43301-0021.flac'),
                           (4, '5678/43303/5678-43303-0024.flac'),
                           (4, '5678/43303/5678-43303-0032.flac'),
                           (5, '6476/57446/6476-57446-0019.flac')]

        # We do not expect the findAllSeqs function to retrieve all sequences
        # in a specific order. However, it should retrieve them all correctly

        # Check the number of speakers
        eq_(len(speakers), 6)

        # Check the speakers names
        eq_(set(speakers), {'2911', '4051', '4397', '5393', '5678', '6476'})

        # Check that all speakers from 0 to 5 are represented
        speaker_set = {x[0] for x in seq_names}
        eq_(speaker_set, {x[0] for x in expected_output})

        # Check the number of sequences
        eq_(len(seq_names), len(expected_output))

        # Check that the sequences are correct
        sequence_set = {x[1] for x in seq_names}
        eq_(sequence_set, {x[1] for x in expected_output})

        # Check that the speakers are properly matched
        for index_speaker, seq_name in seq_names:
            speaker_name = str(Path(seq_name).stem).split('-')[0]
            eq_(speakers[index_speaker], speaker_name)

    def testFindAllSeqsCustomSpeakers(self):
        seq_names, speakers = findAllSeqs(str(self.path_db),
                                          extension=".flac",
                                          speaker_level=2)
        expected_speakers = {'2911/12359', '4051/11218', '4397/15668',
                             '5393/19218', '5678/43301', '5678/43303',
                             '6476/57446'}
        eq_(set(speakers), expected_speakers)

        for index_speaker, seq_name in seq_names:
            speaker_name = '/'.join(str(Path(seq_name).stem).split('-')[:2])
            eq_(speakers[index_speaker], speaker_name)

        expected_output = [(0, '2911/12359/2911-12359-0007.flac'),
                           (1, '4051/11218/4051-11218-0044.flac'),
                           (2, '4397/15668/4397-15668-0003.flac'),
                           (2, '4397/15668/4397-15668-0007.flac'),
                           (3, '5393/19218/5393-19218-0024.flac'),
                           (4, '5678/43301/5678-43301-0021.flac'),
                           (5, '5678/43303/5678-43303-0024.flac'),
                           (5, '5678/43303/5678-43303-0032.flac'),
                           (6, '6476/57446/6476-57446-0019.flac')]

        # Check that the sequences are correct
        sequence_set = {x[1] for x in seq_names}
        eq_(sequence_set, {x[1] for x in expected_output})

    def testFindAllSeqs0Speakers(self):
        seq_names, speakers = findAllSeqs(str(self.path_db / '2911/12359/'),
                                          extension=".flac")
        eq_(speakers, [''])

    def testFindAllSeqs0SpeakersForced(self):
        seq_names, speakers = findAllSeqs(str(self.path_db),
                                          extension=".flac", speaker_level=0)
        eq_(speakers, [''])

    def testLoadData(self):

        seq_names, speakers = findAllSeqs(str(self.path_db),
                                          extension=".flac")
        seq_names = filterSeqs(self.seq_list, seq_names)

        expected_output = [(2, '4397/15668/4397-15668-0003.flac'),
                           (2, '4397/15668/4397-15668-0007.flac'),
                           (3, '5393/19218/5393-19218-0024.flac'),
                           (4, '5678/43301/5678-43301-0021.flac'),
                           (4, '5678/43303/5678-43303-0024.flac'),
                           (4, '5678/43303/5678-43303-0032.flac'),
                           (5, '6476/57446/6476-57446-0019.flac')]

        eq_(len(seq_names), len(expected_output))
        eq_({x[1] for x in seq_names}, {x[1] for x in expected_output})
        phone_labels_dict = None
        n_speakers = 9
        test_data = AudioBatchData(self.path_db, self.size_window,
                                   seq_names, phone_labels_dict, n_speakers)
        assert(test_data.getNSpeakers() == 9)
        assert(test_data.getNSeqs() == 7)

    def testDataLoader(self):

        batch_size = 2
        seq_names, speakers = findAllSeqs(str(self.path_db),
                                          extension=".flac")
        seq_names = filterSeqs(self.seq_list, seq_names)

        test_data = AudioBatchData(self.path_db, self.size_window, seq_names,
                                   None, len(speakers))

        test_data_loader = test_data.getDataLoader(batch_size, "samespeaker",
                                                   True, numWorkers=2)
        visted_labels = set()
        for index, item in enumerate(test_data_loader):
            _, labels = item
            p = labels[0].item()
            visted_labels.add(p)
            eq_(torch.sum(labels == p), labels.size(0))

        eq_(len(visted_labels), 4)

    def testPartialLoader(self):

        batch_size = 16
        seq_names, speakers = findAllSeqs(str(self.path_db),
                                          extension=".flac")
        seq_names = filterSeqs(self.seq_list, seq_names)
        test_data = AudioBatchData(self.path_db, self.size_window,
                                   seq_names, None, len(speakers),
                                   MAX_SIZE_LOADED=1000000)
        eq_(test_data.getNPacks(), 2)
        test_data_loader = test_data.getDataLoader(batch_size, "samespeaker",
                                                   True, numWorkers=2)
        visted_labels = set()
        for index, item in enumerate(test_data_loader):
            _, labels = item
            p = labels[0].item()
            eq_(torch.sum(labels == p), labels.size(0))
            visted_labels.add(p)

        eq_(len(visted_labels), 4)


class TestPhonemParser(unittest.TestCase):

    def setUp(self):
        from .train import parseSeqLabels
        self.seqLoader = parseSeqLabels
        self.test_data_dir = Path(__file__).parent / 'test_data'
        self.pathPhone = self.test_data_dir / 'phone_labels.txt'
        self.path_db = self.test_data_dir / 'test_db'

    def testSeqLoader(self):
        phone_data, nPhones = self.seqLoader(self.pathPhone)
        eq_(len(phone_data), 7)
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
        test_data = AudioBatchData(
            self.path_db, size_window, seq_names, phone_data, len(speakers))
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
