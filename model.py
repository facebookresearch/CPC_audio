import torch
import torch.nn as nn
import torch.nn.functional as F

###########################################
# Networks
###########################################


class EncoderNetwork(nn.Module):

    def __init__(self,
                 sizeHidden=512):

        super(EncoderNetwork, self).__init__()
        self.conv0 = nn.Conv1d(1, sizeHidden, 10, stride=5, padding=3)
        self.batchNorm0 = nn.BatchNorm1d(sizeHidden)
        self.conv1 = nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4, padding=2)
        self.batchNorm1 = nn.BatchNorm1d(sizeHidden)
        self.conv2 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm1d(sizeHidden)
        self.conv3 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm1d(sizeHidden)
        self.conv4 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm1d(sizeHidden)

    def getDimOutput(self):

        return self.conv4.out_channels

    def forward(self, x):

        x = F.relu(self.batchNorm0(self.conv0(x)))
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        x = F.relu(self.batchNorm4(self.conv4(x)))

        return x


class AutoregressiveNetwork(nn.Module):

    def __init__(self,
                 dimEncoded,
                 dimOutput):

        super(AutoregressiveNetwork, self).__init__()

        self.baseNet = nn.GRU(dimEncoded, dimOutput, num_layers=1, batch_first=True)

    def getDimOutput(self):
        return self.baseNet.hidden_size

    def forward(self, x):

        self.baseNet.flatten_parameters()
        return self.baseNet(x)[0]

###########################################
# Model
###########################################


class CPCModel(nn.Module):

    def __init__(self,
                 dimEncoded,
                 dimAR):

        super(CPCModel, self).__init__()
        self.gEncoder = EncoderNetwork(dimEncoded)
        self.gAR = AutoregressiveNetwork(dimEncoded, dimAR)

    def forward(self, batchData, nAR=0):
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)

        dimEncoded = self.gEncoder.getDimOutput()

        if nAR == 0:
            return encodedData

        if nAR == -1:
            nAR = encodedData.size(0)

        gtPredictions = encodedData[:nAR].view(nAR, -1, dimEncoded)
        otherEncoded = encodedData[nAR:].contiguous().view(-1, dimEncoded)

        # We are going to perform one prediction sequence per GPU
        cFeature = self.gAR(gtPredictions)

        return cFeature, gtPredictions, otherEncoded
