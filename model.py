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
                 dimOutput,
                 keepHidden):

        super(AutoregressiveNetwork, self).__init__()

        self.baseNet = nn.GRU(dimEncoded, dimOutput,
                              num_layers=1, batch_first=True)
        self.hidden = None
        self.keepHidden = keepHidden

    def getDimOutput(self):
        return self.baseNet.hidden_size

    def forward(self, x):

        self.baseNet.flatten_parameters()
        x, h = self.baseNet(x, self.hidden)
        if self.keepHidden:
            self.hidden = h.detach()
        return x

###########################################
# Model
###########################################


class CPCModel(nn.Module):

    def __init__(self,
                 dimEncoded,
                 dimAR,
                 keepHidden):

        super(CPCModel, self).__init__()
        self.gEncoder = EncoderNetwork(dimEncoded)
        self.gAR = AutoregressiveNetwork(dimEncoded, dimAR, keepHidden)

    def forward(self, batchData):
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)

        # We are going to perform one prediction sequence per GPU
        cFeature = self.gAR(encodedData)

        return cFeature, encodedData


class ID(nn.Module):

    def __init__(self):
        super(ID, self).__init__()

    def forward(self, x, *args, **kwargs):
        return x, None, None
