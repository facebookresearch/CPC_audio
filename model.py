import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import torch

###########################################
# Networks
###########################################


class CPCEncoder(nn.Module):

    def __init__(self,
                 sizeHidden=512):

        super(CPCEncoder, self).__init__()
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
        self.DOWNSAMPLING = 160

    def getDimOutput(self):
        return self.conv4.out_channels

    def forward(self, x):
        x = F.relu(self.batchNorm0(self.conv0(x)))
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        x = F.relu(self.batchNorm4(self.conv4(x)))

        return x


class MFCCEncoder(nn.Module):

    def __init__(self,
                 dimEncoded):

        super(MFCCEncoder, self).__init__()
        melkwargs = {"n_mels": max(128, dimEncoded), "n_fft": 321}
        self.MFCC = torchaudio.transforms.MFCC(n_mfcc=dimEncoded,
                                               melkwargs=melkwargs)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.MFCC(x)
        return x.permute(0, 2, 1)


class LFBEnconder(nn.Module):

    def __init__(self, dimEncoded, normalize=True):

        super(LFBEnconder, self).__init__()
        self.dimEncoded = dimEncoded
        self.conv = nn.Conv1d(1, 2 * dimEncoded,
                              400, stride=1)
        self.register_buffer('han', torch.hann_window(400).view(1, 1, 400))
        self.instancenorm = nn.InstanceNorm1d(dimEncoded, momentum=1) \
            if normalize else None

    def forward(self, x):

        N, C, L = x.size()
        x = self.conv(x)
        x = x.view(N, self.dimEncoded, 2, -1)
        x = x[:, :, 0, :]**2 + x[:, :, 1, :]**2
        x = x.view(N * self.dimEncoded, 1,  -1)
        x = torch.nn.functional.conv1d(x, self.han, bias=None,
                                       stride=160, padding=350)
        x = x.view(N, self.dimEncoded,  -1)
        x = torch.log(1 + torch.abs(x))

        # Normalization
        if self.instancenorm is not None:
            x = self.instancenorm(x)
        return x


class CPCAR(nn.Module):

    def __init__(self,
                 dimEncoded,
                 dimOutput,
                 keepHidden,
                 nLevelsGRU,
                 reverse=False):

        super(CPCAR, self).__init__()

        self.baseNet = nn.GRU(dimEncoded, dimOutput,
                              num_layers=nLevelsGRU, batch_first=True)
        self.hidden = None
        self.keepHidden = keepHidden
        self.reverse = reverse

    def getDimOutput(self):
        return self.baseNet.hidden_size

    def forward(self, x):

        if self.reverse:
            x = torch.flip(x, [1])
        self.baseNet.flatten_parameters()
        x, h = self.baseNet(x, self.hidden)
        if self.keepHidden:
            self.hidden = h.detach()

        # For better modularity, a sequence's order should be preserved
        # by each module
        if self.reverse:
            x = torch.flip(x, [1])
        return x

class NoAr(nn.Module):

     def __init__(self, *args):
        super(NoAr, self).__init__()

     def forward(self, x):
        return x


class BiDIRARTangled(nn.Module):

    def __init__(self,
                 dimEncoded,
                 dimOutput,
                 nLevelsGRU):

        super(BiDIRARTangled, self).__init__()
        assert(dimOutput % 2 == 0)

        self.ARNet = nn.GRU(dimEncoded, dimOutput // 2,
                            num_layers=nLevelsGRU, batch_first=True,
                            bidirectional=True)

    def getDimOutput(self):
        return self.ARNet.hidden_size * 2

    def forward(self, x):

        self.ARNet.flatten_parameters()
        xf, _ = self.ARNet(x)
        return xf


class BiDIRAR(nn.Module):

    def __init__(self,
                 dimEncoded,
                 dimOutput,
                 nLevelsGRU):

        super(BiDIRAR, self).__init__()
        assert(dimOutput % 2 == 0)

        self.netForward = nn.GRU(dimEncoded, dimOutput // 2,
                                 num_layers=nLevelsGRU, batch_first=True)
        self.netBackward = nn.GRU(dimEncoded, dimOutput // 2,
                                  num_layers=nLevelsGRU, batch_first=True)

    def getDimOutput(self):
        return self.netForward.hidden_size * 2

    def forward(self, x):

        self.netForward.flatten_parameters()
        self.netBackward.flatten_parameters()
        xf, _ = self.netForward(x)
        xb, _ = self.netBackward(torch.flip(x, [1]))
        return torch.cat([xf, torch.flip(xb, [1])], dim=2)


###########################################
# Model
###########################################


class CPCModel(nn.Module):

    def __init__(self,
                 encoder,
                 AR):

        super(CPCModel, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR

    def forward(self, batchData, label):
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        cFeature = self.gAR(encodedData)
        return cFeature, encodedData, label


class CPCBertModel(nn.Module):

    def __init__(self,
                 encoder,
                 AR,
                 nMaskSentence=2,
                 blockSize=12):

        super(CPCBertModel, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR
        self.blockSize = blockSize
        self.nMaskSentence = nMaskSentence
        self.supervised = False

    def getMask(self, batchData):

        batchSize, seqSize, c = batchData.size()
        maskLabel = torch.randint(0, seqSize // self.blockSize,
                                  (self.nMaskSentence * batchSize, 1))
        maskLabel *= self.blockSize

        baseX = torch.arange(0, self.blockSize, dtype=torch.long)
        baseX = baseX.expand(self.nMaskSentence * batchSize, self.blockSize)
        maskLabel = maskLabel + baseX
        maskLabel = maskLabel.view(-1)

        baseY = torch.arange(0, batchSize,
                             dtype=torch.long).view(-1, 1) * seqSize
        baseY = baseY.expand(batchSize,
                             self.nMaskSentence *
                             self.blockSize).contiguous().view(-1)
        maskLabel = maskLabel + baseY
        outLabels = torch.zeros(batchSize * seqSize,
                                dtype=torch.uint8)
        outLabels[maskLabel] = 1

        outLabels = outLabels.view(batchSize, seqSize)

        return outLabels

    def forward(self, batchData, label):

        fullEncoded = self.gEncoder(batchData).permute(0, 2, 1)

        # Sample random blocks of data
        if not self.supervised:
            maskLabels = self.getMask(fullEncoded)
            partialEncoded = fullEncoded.clone()
            partialEncoded[maskLabels] = 0
            cFeature = self.gAR(partialEncoded)
            return cFeature, fullEncoded, maskLabels.cuda()

        else:
            cFeature = self.gAR(fullEncoded)
            return cFeature, fullEncoded, label



class ConcatenatedModel(nn.Module):

    def __init__(self, model_list):

        super(ConcatenatedModel, self).__init__()
        self.models = torch.nn.ModuleList(model_list)

    def forward(self, batchData, label):

        outFeatures = []
        outEncoded = []
        for model in self.models:
            cFeature, encodedData = model(batchData)
            outFeatures.append(cFeature)
            outEncoded.append(encodedData)
        return torch.cat(outFeatures, dim=2), torch.cat(outEncoded, dim=2), label
