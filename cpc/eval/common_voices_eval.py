# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os
import torchaudio
from copy import deepcopy
import torch
import time
import random
import math
import json
import subprocess
import sys
import progressbar
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.multiprocessing import Pool
from cpc.criterion.seq_alignment import get_seq_PER
from cpc.criterion.seq_alignment import beam_search
from cpc.feature_loader import loadModel
from cpc.dataset import findAllSeqs, parseSeqLabels, filterSeqs


def load(path_item):
    seq_name = path_item.stem
    data = torchaudio.load(str(path_item))[0].view(1, -1)
    return seq_name, data


class SingleSequenceDataset(Dataset):

    def __init__(self,
                 pathDB,
                 seqNames,
                 phoneLabelsDict,
                 inDim=1,
                 transpose=True):
        """
        Args:
            - path (string): path to the training dataset
            - sizeWindow (int): size of the sliding window
            - seqNames (list): sequences to load
            - phoneLabels (dictionnary): if not None, a dictionnary with the
                                         following entries

                                         "step": size of a labelled window
                                         "$SEQ_NAME": list of phonem labels for
                                         the sequence $SEQ_NAME
        """
        self.seqNames = deepcopy(seqNames)
        self.pathDB = pathDB
        self.phoneLabelsDict = deepcopy(phoneLabelsDict)
        self.inDim = inDim
        self.transpose = transpose
        self.loadSeqs()

    def loadSeqs(self):

        # Labels
        self.seqOffset = [0]
        self.phoneLabels = []
        self.phoneOffsets = [0]
        self.data = []
        self.maxSize = 0
        self.maxSizePhone = 0

        # Data

        nprocess = min(30, len(self.seqNames))

        start_time = time.time()
        to_load = [Path(self.pathDB) / x for _, x in self.seqNames]

        with Pool(nprocess) as p:
            poolData = p.map(load, to_load)

        tmpData = []
        poolData.sort()

        totSize = 0
        minSizePhone = float('inf')
        for seqName, seq in poolData:
            self.phoneLabels += self.phoneLabelsDict[seqName]
            self.phoneOffsets.append(len(self.phoneLabels))
            self.maxSizePhone = max(self.maxSizePhone, len(
                self.phoneLabelsDict[seqName]))
            minSizePhone = min(minSizePhone, len(
                self.phoneLabelsDict[seqName]))
            sizeSeq = seq.size(1)
            self.maxSize = max(self.maxSize, sizeSeq)
            totSize += sizeSeq
            tmpData.append(seq)
            self.seqOffset.append(self.seqOffset[-1] + sizeSeq)
            del seq
        self.data = torch.cat(tmpData, dim=1)
        self.phoneLabels = torch.tensor(self.phoneLabels, dtype=torch.long)
        print(f'Loaded {len(self.phoneOffsets)} sequences '
              f'in {time.time() - start_time:.2f} seconds')
        print(f'maxSizeSeq : {self.maxSize}')
        print(f'maxSizePhone : {self.maxSizePhone}')
        print(f"minSizePhone : {minSizePhone}")
        print(f'Total size dataset {totSize / (16000 * 3600)} hours')

    def __getitem__(self, idx):

        offsetStart = self.seqOffset[idx]
        offsetEnd = self.seqOffset[idx+1]
        offsetPhoneStart = self.phoneOffsets[idx]
        offsetPhoneEnd = self.phoneOffsets[idx + 1]

        sizeSeq = int(offsetEnd - offsetStart)
        sizePhone = int(offsetPhoneEnd - offsetPhoneStart)

        outSeq = torch.zeros((self.inDim, self.maxSize))
        outPhone = torch.zeros((self.maxSizePhone))

        outSeq[:, :sizeSeq] = self.data[:, offsetStart:offsetEnd]
        outPhone[:sizePhone] = self.phoneLabels[offsetPhoneStart:offsetPhoneEnd]

        return outSeq,  torch.tensor([sizeSeq], dtype=torch.long), outPhone.long(),  torch.tensor([sizePhone], dtype=torch.long)

    def __len__(self):
        return len(self.seqOffset) - 1


class CTCphone_criterion(torch.nn.Module):

    def __init__(self, dimEncoder, nPhones, LSTM=False, sizeKernel=8,
                 seqNorm=False, dropout=False, reduction='sum'):

        super(CTCphone_criterion, self).__init__()
        self.seqNorm = seqNorm
        self.epsilon = 1e-8
        self.dropout = torch.nn.Dropout2d(
            p=0.5, inplace=False) if dropout else None
        self.conv1 = torch.nn.LSTM(dimEncoder, dimEncoder,
                                   num_layers=1, batch_first=True)
        self.PhoneCriterionClassifier = torch.nn.Conv1d(
            dimEncoder, nPhones + 1, sizeKernel, stride=sizeKernel // 2)
        self.lossCriterion = torch.nn.CTCLoss(blank=nPhones,
                                              reduction=reduction,
                                              zero_infinity=True)
        self.relu = torch.nn.ReLU()
        self.BLANK_LABEL = nPhones
        self.useLSTM = LSTM

    def getPrediction(self, cFeature, featureSize):
        B, S, H = cFeature.size()
        if self.seqNorm:
            for b in range(B):
                size = featureSize[b]
                m = cFeature[b, :size].mean(dim=0, keepdim=True)
                v = cFeature[b, :size].var(dim=0, keepdim=True)
                cFeature[b] = (cFeature[b] - m) / torch.sqrt(v + self.epsilon)
        if self.useLSTM:
            cFeature = self.conv1(cFeature)[0]

        cFeature = cFeature.permute(0, 2, 1)

        if self.dropout is not None:
            cFeature = self.dropout(cFeature)

        cFeature = self.PhoneCriterionClassifier(cFeature)
        return cFeature.permute(0, 2, 1)

    def forward(self, cFeature, featureSize, label, labelSize):

        # cFeature.size() : batchSize x seq Size x hidden size
        B, S, H = cFeature.size()
        predictions = self.getPrediction(cFeature, featureSize)
        featureSize /= 4
        predictions = cut_data(predictions, featureSize)
        featureSize = torch.clamp(featureSize, max=predictions.size(1))
        label = cut_data(label, labelSize)

        if labelSize.min() <= 0:
            print(label, labelSize)
        predictions = torch.nn.functional.log_softmax(predictions, dim=2)
        predictions = predictions.permute(1, 0, 2)
        loss = self.lossCriterion(predictions, label,
                                  featureSize, labelSize).view(1, -1)

        if torch.isinf(loss).sum() > 0 or torch.isnan(loss).sum() > 0:
            loss = 0

        return loss


class IDModule(torch.nn.Module):

    def __init__(self):
        super(IDModule, self).__init__()

    def forward(self, feature, *args):
        B, C, S = feature.size()
        return feature.permute(0, 2, 1), None, None


def cut_data(seq, sizeSeq):
    maxSeq = sizeSeq.max()
    return seq[:, :maxSeq]


def prepare_data(data):
    seq, sizeSeq, phone, sizePhone = data
    seq = seq.cuda(non_blocking=True)
    phone = phone.cuda(non_blocking=True)
    sizeSeq = sizeSeq.cuda(non_blocking=True).view(-1)
    sizePhone = sizePhone.cuda(non_blocking=True).view(-1)

    seq = cut_data(seq.permute(0, 2, 1), sizeSeq).permute(0, 2, 1)

    return seq, sizeSeq, phone, sizePhone


def train_step(train_loader,
               model,
               criterion,
               optimizer,
               downsampling_factor):

    if model.optimize:
        model.train()

    criterion.train()
    avg_loss = 0
    nItems = 0

    for data in train_loader:
        optimizer.zero_grad()
        seq, sizeSeq, phone, sizePhone = prepare_data(data)
        c_feature, _, _ = model(seq, None)
        if not model.optimize:
            c_feature = c_feature.detach()
        sizeSeq = sizeSeq / downsampling_factor
        loss = criterion(c_feature, sizeSeq, phone, sizePhone)
        loss.mean().backward()

        avg_loss += loss.mean().item()
        nItems += 1
        optimizer.step()

    return avg_loss / nItems


def val_step(val_loader,
             model,
             criterion,
             downsampling_factor):

    model.eval()
    criterion.eval()
    avg_loss = 0
    nItems = 0

    for data in val_loader:
        with torch.no_grad():
            seq, sizeSeq, phone, sizePhone = prepare_data(data)
            c_feature, _, _ = model(seq, None)
            sizeSeq = sizeSeq / downsampling_factor
            loss = criterion(c_feature, sizeSeq, phone, sizePhone)
            avg_loss += loss.mean().item()
            nItems += 1

    return avg_loss / nItems


def get_per(data):
    pred, size_pred, gt, size_gt, blank_label = data
    l_ = min(size_pred // 4, pred.size(0))
    p_ = pred[:l_].view(l_, -1).numpy()
    gt_seq = gt[:size_gt].view(-1).tolist()
    predSeq = beam_search(p_, 20, blank_label)[0][1]
    out = get_seq_PER(gt_seq, predSeq)
    return out


def perStep(val_loader,
            model,
            criterion,
            downsampling_factor):

    model.eval()
    criterion.eval()

    avgPER = 0
    varPER = 0
    nItems = 0

    print("Starting the PER computation through beam search")
    bar = progressbar.ProgressBar(maxval=len(val_loader))
    bar.start()

    for index, data in enumerate(val_loader):

        bar.update(index)

        with torch.no_grad():
            seq, sizeSeq, phone, sizePhone = prepare_data(data)
            c_feature, _, _ = model(seq, None)
            sizeSeq = sizeSeq / downsampling_factor
            predictions = torch.nn.functional.softmax(
                criterion.module.getPrediction(c_feature, sizeSeq), dim=2).cpu()
            phone = phone.cpu()
            sizeSeq = sizeSeq.cpu()
            sizePhone = sizePhone.cpu()

            bs = c_feature.size(0)
            data_per = [(predictions[b], sizeSeq[b], phone[b], sizePhone[b],
                         criterion.module.BLANK_LABEL) for b in range(bs)]

            with Pool(bs) as p:
                poolData = p.map(get_per, data_per)
            avgPER += sum([x for x in poolData])
            varPER += sum([x*x for x in poolData])
            nItems += len(poolData)

    bar.finish()

    avgPER /= nItems
    varPER /= nItems

    varPER -= avgPER**2
    print(f"Average PER {avgPER}")
    print(f"Standard deviation PER {math.sqrt(varPER)}")


def run(train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        downsampling_factor,
        nEpochs,
        pathCheckpoint):

    print(f"Starting the training for {nEpochs} epochs")
    bestLoss = float('inf')

    for epoch in range(nEpochs):
        lossTrain = train_step(train_loader, model, criterion,
                               optimizer, downsampling_factor)

        print(f"Epoch {epoch} loss train : {lossTrain}")

        lossVal = val_step(val_loader, model, criterion, downsampling_factor)
        print(f"Epoch {epoch} loss val : {lossVal}")

        if lossVal < bestLoss:
            bestLoss = lossVal
            state_dict = {'classifier': criterion.state_dict(),
                          'model': model.state_dict(),
                          'bestLoss': bestLoss}
            torch.save(state_dict, pathCheckpoint)


def get_PER_args(args):

    path_args_training = os.path.join(args.output, "args_training.json")
    with open(path_args_training, 'rb') as file:
        data = json.load(file)

    if args.pathDB is None:
        args.pathDB = data["pathDB"]
        args.file_extension = data["file_extension"]

    if args.pathVal is None and args.pathPhone is None:
        args.pathPhone = data["pathPhone"]
        args.pathVal = data["pathVal"]

    args.pathCheckpoint = data["pathCheckpoint"]
    args.no_pretraining = data["no_pretraining"]
    args.LSTM = data.get("LSTM", False)
    args.seqNorm = data.get("seqNorm", False)
    args.dropout = data.get("dropout", False)
    args.in_dim = data.get("in_dim", 1)
    args.loss_reduction = data.get("loss_reduction", "mean")
    return args


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='Simple phone recognition pipeline '
                                                 'for the common voices datasets')

    subparsers = parser.add_subparsers(dest='command')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('pathDB', type=str,
                              help='Path to the directory containing the '
                              'audio data / pre-computed features.')
    parser_train.add_argument('pathPhone', type=str,
                              help='Path to the .txt file containing the '
                              'phone transcription.')
    parser_train.add_argument('pathCheckpoint', type=str,
                              help='Path to the CPC checkpoint to load. '
                              'Set to ID to work with pre-cimputed features.')
    parser_train.add_argument('--freeze', action='store_true',
                              help="Freeze the CPC features layers")
    parser_train.add_argument('--pathTrain', default=None, type=str,
                              help='Path to the .txt files containing the '
                              'list of the training sequences.')
    parser_train.add_argument('--pathVal', default=None, type=str,
                              help='Path to the .txt files containing the '
                              'list of the validation sequences.')
    parser_train.add_argument('--file_extension', type=str, default=".mp3",
                              help='Extension of the files in the '
                              'dataset')
    parser_train.add_argument('--batchSize', type=int, default=8)
    parser_train.add_argument('--nEpochs', type=int, default=30)
    parser_train.add_argument('--beta1', type=float, default=0.9,
                              help='Value of beta1 for the Adam optimizer.')
    parser_train.add_argument('--beta2', type=float, default=0.999,
                              help='Value of beta2 for the Adam optimizer.')
    parser_train.add_argument('--epsilon', type=float, default=1e-08,
                              help='Value of epsilon for the Adam optimizer.')
    parser_train.add_argument('--lr', type=float, default=2e-04,
                              help='Learning rate.')
    parser_train.add_argument('-o', '--output', type=str, default='out',
                              help="Output directory")
    parser_train.add_argument('--debug', action='store_true',
                              help='If activated, will only load a few '
                              'sequences from the dataset.')
    parser_train.add_argument('--no_pretraining', action='store_true',
                              help='Activate use a randmly initialized '
                              'network')
    parser_train.add_argument('--LSTM', action='store_true',
                              help='Activate to add a LSTM to the phone '
                              'classifier')
    parser_train.add_argument('--seqNorm', action='store_true',
                              help='Activate if you want to normalize each '
                              'batch of features through time before the '
                              'phone classification.')
    parser_train.add_argument('--kernelSize', type=int, default=8,
                              help='Number of features to concatenate before '
                              'feeding them to the phone classifier.')
    parser_train.add_argument('--dropout', action='store_true')
    parser_train.add_argument('--in_dim', type=int, default=1,
                              help='Dimension of the input data: useful when '
                              'working with pre-computed features or '
                              'stereo audio.')
    parser_train.add_argument('--loss_reduction', type=str, default='mean',
                              choices=['mean', 'sum'])

    parser_per = subparsers.add_parser('per')
    parser_per.add_argument('output', type=str)
    parser_per.add_argument('--batchSize', type=int, default=8)
    parser_per.add_argument('--debug', action='store_true',
                            help='If activated, will only load a few '
                            'sequences from the dataset.')
    parser_per.add_argument('--pathDB',
                            help="For computing the PER on another dataset",
                            type=str, default=None)
    parser_per.add_argument('--pathVal',
                            help="For computing the PER on specific sequences",
                            type=str, default=None)
    parser_per.add_argument('--pathPhone',
                            help="For computing the PER on specific sequences",
                            default=None, type=str)
    parser_per.add_argument('--file_extension', type=str, default=".mp3")
    parser_per.add_argument('--name', type=str, default="0")

    args = parser.parse_args()

    if args.command == 'per':
        args = get_PER_args(args)

    # Output Directory
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    name = f"_{args.name}" if args.command == "per" else ""
    pathLogs = os.path.join(args.output, f'logs_{args.command}{name}.txt')
    tee = subprocess.Popen(["tee", pathLogs], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())

    phoneLabels, nPhones = parseSeqLabels(args.pathPhone)

    inSeqs, _ = findAllSeqs(args.pathDB,
                            extension=args.file_extension)
    # Datasets
    if args.command == 'train' and args.pathTrain is not None:
        seqTrain = filterSeqs(args.pathTrain, inSeqs)
    else:
        seqTrain = inSeqs

    if args.pathVal is None and args.command == 'train':
        random.shuffle(seqTrain)
        sizeTrain = int(0.9 * len(seqTrain))
        seqTrain, seqVal = seqTrain[:sizeTrain], seqTrain[sizeTrain:]
    elif args.pathVal is not None:
        seqVal = filterSeqs(args.pathVal, inSeqs)
    else:
        raise RuntimeError("No validation dataset found for PER computation")

    if args.debug:
        seqVal = seqVal[:100]

    downsampling_factor = 160
    if args.pathCheckpoint == 'ID':
        downsampling_factor = 1
        feature_maker = IDModule()
        hiddenGar = args.in_dim
    else:
        feature_maker, hiddenGar, _ = loadModel([args.pathCheckpoint],
                                                loadStateDict=not args.no_pretraining)
    feature_maker.cuda()
    feature_maker = torch.nn.DataParallel(feature_maker)

    phone_criterion = CTCphone_criterion(hiddenGar, nPhones, args.LSTM,
                                         seqNorm=args.seqNorm,
                                         dropout=args.dropout,
                                         reduction=args.loss_reduction)
    phone_criterion.cuda()
    phone_criterion = torch.nn.DataParallel(phone_criterion)

    print(f"Loading the validation dataset at {args.pathDB}")
    datasetVal = SingleSequenceDataset(args.pathDB, seqVal,
                                       phoneLabels, inDim=args.in_dim)

    val_loader = DataLoader(datasetVal, batch_size=args.batchSize,
                            shuffle=True)

    # Checkpoint file where the model should be saved
    pathCheckpoint = os.path.join(args.output, 'checkpoint.pt')

    if args.command == 'train':
        feature_maker.optimize = True
        if args.freeze:
            feature_maker.eval()
            feature_maker.optimize = False
            for g in feature_maker.parameters():
                g.requires_grad = False

        if args.debug:
            print("debug")
            random.shuffle(seqTrain)
            seqTrain = seqTrain[:1000]
            seqVal = seqVal[:100]

        print(f"Loading the training dataset at {args.pathDB}")

        datasetTrain = SingleSequenceDataset(args.pathDB, seqTrain,
                                             phoneLabels, inDim=args.in_dim)

        train_loader = DataLoader(datasetTrain, batch_size=args.batchSize,
                                  shuffle=True)

        # Optimizer
        g_params = list(phone_criterion.parameters())
        if not args.freeze:
            print("Optimizing model")
            g_params += list(feature_maker.parameters())

        optimizer = torch.optim.Adam(g_params, lr=args.lr,
                                     betas=(args.beta1, args.beta2),
                                     eps=args.epsilon)

        pathArgs = os.path.join(args.output, "args_training.json")
        with open(pathArgs, 'w') as file:
            json.dump(vars(args), file, indent=2)

        run(train_loader, val_loader, feature_maker, phone_criterion,
            optimizer, downsampling_factor, args.nEpochs, pathCheckpoint)

    else:
        print(f"Loading data at {pathCheckpoint}")
        state_dict = torch.load(pathCheckpoint,
                                map_location=lambda storage, loc: storage)
        if 'bestLoss' in state_dict:
            print(f"Best loss : {state_dict['bestLoss']}")
        phone_criterion.load_state_dict(state_dict['classifier'])
        feature_maker.load_state_dict(state_dict['model'])

        pathArgs = os.path.join(args.output,
                                f"args_validation_{args.name}.json")
        with open(pathArgs, 'w') as file:
            json.dump(vars(args), file, indent=2)

        perStep(val_loader,
                feature_maker,
                phone_criterion,
                downsampling_factor)
