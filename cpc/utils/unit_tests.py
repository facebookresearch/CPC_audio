# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import unittest
import torch
import os
from nose.tools import eq_, ok_

from .misc import SchedulerCombiner, ramp_scheduling_function


class TestCombineSchedulers(unittest.TestCase):

    def setUp(self):
        self.baseLR = 1
        self.module = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(
            list(self.module.parameters()), lr=self.baseLR)

    def testCombineRamp(self):
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                      lr_lambda=lambda epoch: ramp_scheduling_function(
                                                          3, epoch))
        self.optimizer.step()
        eq_(self.optimizer.param_groups[0]['lr'], self.baseLR / 3)
        scheduler.step()
        eq_(self.optimizer.param_groups[0]['lr'], 2 * self.baseLR / 3)
        scheduler.step()
        eq_(self.optimizer.param_groups[0]['lr'], 1)

        for i in range(12):
            scheduler.step()
            eq_(self.optimizer.param_groups[0]['lr'], 1)

    def testCombineRampStep(self):
        scheduler_step = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 6, gamma=0.5)
        scheduler_ramp = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                           lr_lambda=lambda epoch: ramp_scheduling_function(
                                                               3, epoch))

        scheduler = SchedulerCombiner([scheduler_ramp, scheduler_step], [0, 3])
        self.optimizer.step()
        # Epoch 0
        eq_(self.optimizer.param_groups[0]['lr'], self.baseLR / 3)
        scheduler.step()
        # Epoch 1
        eq_(self.optimizer.param_groups[0]['lr'], 2 * self.baseLR / 3)
        scheduler.step()
        # Epoch 2
        eq_(self.optimizer.param_groups[0]['lr'], 1)
        scheduler.step()

        # Epoch 3, 4, 5
        for i in range(3):
            eq_(self.optimizer.param_groups[0]['lr'], 1)
            scheduler.step()

        # Epoch 6
        eq_(self.optimizer.param_groups[0]['lr'], 0.5)
