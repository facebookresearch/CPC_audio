import torch
import torchaudio
import os


def toOneHot(inputVector, nItems):

    batchSize, seqSize = inputVector.size()
    out = torch.zeros((batchSize, seqSize, nItems),
                      device=inputVector.device, dtype=torch.long)
    out.scatter_(2, inputVector.view(batchSize, seqSize, 1), 1)
    return out


def loadCriterion(pathCheckpoint):
    from criterion import PhoneCriterion, CTCPhoneCriterion
    from train import parseSeqLabels, getCheckpointData

    *_, args = getCheckpointData(os.path.dirname(pathCheckpoint))
    _, nPhones = parseSeqLabels(args.pathPhone)
    if args.CTC:
        criterion = CTCPhoneCriterion(args.hiddenGar if not args.onEncoder \
                                      else args.hiddenEncoder,
                                      nPhones, args.onEncoder)
    else:
        criterion = PhoneCriterion(args.hiddenGar, nPhones, args.onEncoder)

    state_dict = torch.load(pathCheckpoint)
    criterion.load_state_dict(state_dict["cpcCriterion"])
    return criterion, nPhones


class FeatureModule(torch.nn.Module):

    def __init__(self, featureMaker, get_encoded, collapse=False):
        super(FeatureModule, self).__init__()
        self.get_encoded = get_encoded
        self.featureMaker = featureMaker
        self.collapse = collapse

    def forward(self, data):

        batchAudio, label = data
        cFeature, encoded, _ = self.featureMaker(batchAudio.cuda(), label)
        if self.get_encoded:
            cFeature = encoded
        if self.collapse:
            cFeature = cFeature.contiguous().view(-1, cFeature.size(2))
        return cFeature


class ModelPhoneCombined(torch.nn.Module):
    def __init__(self, model, criterion, oneHot):
        super(ModelPhoneCombined, self).__init__()
        self.model = model
        self.criterion = criterion
        self.oneHot = oneHot

    def forward(self, data):
        c_feature = self.model(data)
        pred = self.criterion.getPrediction(c_feature)
        P = pred.size(2)

        if self.oneHot:
            pred = pred.argmax(dim=2)
            pred = toOneHot(pred, P)
        else:
            pred = torch.nn.functional.softmax(pred, dim=2)
        return pred


class ModelClusterCombined(torch.nn.Module):
    def __init__(self, model, cluster, nk, outFormat):

        if outFormat not in ['oneHot', 'int', 'softmax']:
            raise ValueError(f'Invalid output format {outFormat}')

        super(ModelClusterCombined, self).__init__()
        self.model = model
        self.cluster = cluster
        self.nk = nk
        self.outFormat = outFormat

    def forward(self, data):
        c_feature = self.model(data)
        pred = self.cluster(c_feature)
        if self.outFormat == 'oneHot':
            pred = pred.min(dim=2)[1]
            pred = toOneHot(pred, self.nk)
        elif self.outFormat == 'int':
            pred = pred.min(dim=2)[1]
        else:
            pred = torch.nn.functional.softmax(-pred, dim=2)
        return pred


def buildFeature(featureMaker, seqPath, strict=False,
                 maxSizeSeq=64000):

    seq = torchaudio.load(seqPath)[0]
    sizeSeq = seq.size(1)
    start = 0
    out = []
    while start < sizeSeq:
        if strict and start + maxSizeSeq > sizeSeq:
            break
        end = min(sizeSeq, start + maxSizeSeq)
        subseq = (seq[:, start:end]).view(1, 1, -1).cuda(device=0)
        with torch.no_grad():
            features = featureMaker((subseq, None))
        out.append(features.detach().cpu())
        start += maxSizeSeq

    if strict and start < sizeSeq:
        subseq = (seq[:, -maxSizeSeq:]).view(1, 1, -1).cuda(device=0)
        with torch.no_grad():
            features = featureMaker((subseq, None))
        out.append(features[:, start:].detach().cpu())

    return torch.cat(out, dim=1)
