# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import unittest
import torch
from nose.tools import eq_, ok_
from .clustering import kMeanCluster
from .dim_reduction import PCA


class TestKMean(unittest.TestCase):

    def setUp(self):
        pass

    def testKMeanStep(self):
        Ck = torch.tensor([[1, 0, 2],
                           [-1, 3, 1]]).float()

        cluster = kMeanCluster(Ck.view(1, 2, 3))
        a = torch.tensor([[0, 0, 1], [0, 0, 0], [1, 1, 1]]).float()
        a = a.view(3, 1, 3)
        norm = cluster(a).view(3, 2)

        eq_(norm[0, 0], 2.)
        eq_(norm[0, 1], 10.)
        eq_(norm[1, 0], 5.)
        eq_(norm[1, 1], 11.)
        eq_(norm[2, 0], 2.)
        eq_(norm[2, 1], 8.)


class TestPCA(unittest.TestCase):

    def setUp(self):
        self.N = 6
        self.k = 3

    def testPCA(self):

        module = PCA(self.k)
        data = torch.tensor([[0.1681, -0.6360, -0.5347],
                             [-1.7113,  0.5962, -0.5708],
                             [-0.4865,  0.7551, -0.0701],
                             [0.4084, -0.2050,  0.5220],
                             [-2.4044, -1.5335, -0.2291],
                             [0.3060,  0.4486, -1.7393]])

        module.update(data[:3])
        module.update(data[3:])

        eq_(module.N, 6)
        ok_((module.mean -
             torch.tensor([-3.7197, -0.5747, -2.6220])).norm() < 1e-3)
        ok_((module.var - torch.tensor([[9.2351,  2.2462,  1.1529],
                                        [2.2462,  3.9251, -0.5890],
                                        [1.1529, -0.5890,  3.9669]])).norm() < 1e-3)

        module.build(normalize=False)
        ref_mean = torch.tensor([-0.6199, -0.0958, -0.4370])

        ok_((module.mean - ref_mean).norm() < 1e-3)
        ok_((module.var - torch.tensor([[1.1548,  0.3150, -0.0788],
                                        [0.3150,  0.6450, -0.1400],
                                        [-0.0788, -0.1400,  0.4702]])).norm() < 1e-3)

        test_item = torch.tensor([[1., 0., 0.], [0.,  0., 1.]]) + ref_mean

        projected_items = module(test_item)
        expected_projection = torch.tensor(
            [[0.1665,  0.4362, -0.8843], [-0.7785,  0.6085,  0.1536]])
        ok_((projected_items - expected_projection).norm() < 1e-3)
