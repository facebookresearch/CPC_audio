# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import numpy as np
from ..criterion import BaseCriterion


class CPCBertCriterion(BaseCriterion):

    def __init__(self,
                 dimOutputAR,           # Dimension of G_ar
                 dimOutputEncoder,      # Dimension of the convolutional net
                 negativeSamplingExt):  # Number of negative samples to draw

        super(CPCBertCriterion, self).__init__()
        self.wPrediction = torch.nn.Linear(dimOutputAR,
                                           dimOutputEncoder,
                                           bias=False)

        self.negativeSamplingExt = negativeSamplingExt
        self.lossCriterion = nn.CrossEntropyLoss()

    def sample(self, encodedData, mask):

        batchSize, nNegativeExt, dimEncoded = encodedData.size()
        negExt = encodedData[1 - mask].view(-1, dimEncoded)
        nPos = mask.sum()
        extIdx = np.random.randint(0, negExt.size(0),
                                   size=(self.negativeSamplingExt * nPos))
        negExt = negExt[extIdx].view(nPos, self.negativeSamplingExt,
                                     dimEncoded)

        posData = encodedData[mask].view(nPos, 1, dimEncoded)
        labelLoss = torch.zeros(nPos, dtype=torch.long,
                                device=encodedData.device)

        return torch.cat([posData, negExt], dim=1), labelLoss

    def forward(self, cFeature, encodedData, label):

        batchSize, seqSize, dimAR = cFeature.size()
        samples, labelLoss = self.sample(encodedData, label)
        nPos = labelLoss.size(0)
        predictions = self.wPrediction(cFeature[label]).view(nPos, 1, -1)
        predictions = (samples * predictions).mean(dim=2)
        loss = self.lossCriterion(predictions, labelLoss)
        _, predsIndex = predictions.max(1)
        acc = torch.sum(predsIndex == labelLoss).double(
        ).view(1, -1) / nPos

        return loss.view(1, 1), acc
