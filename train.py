import argparse
import json
import os
import random
from termcolor import colored

import numpy as np
import torch

from dataset import AudioBatchData, findAllSeqs, filterSeqs
from model import CPCModel, ConcatenatedModel
from criterion import CPCUnsupersivedCriterion, SpeakerCriterion, \
    PhoneCriterion, ModelCriterionCombined
import psutil


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def updateAndShowLogs(text, logs, nPredicts):
    logStep = logs["step"]
    print("")
    print('-'*50)
    print(text)
    strSteps = ['Step'] + [str(s) for s in range(1, nPredicts + 1)]
    formatCommand = ' '.join(['{:>16}' for x in range(nPredicts + 1)])
    print(formatCommand.format(*strSteps))

    for key in logs:

        if key == "step":
            continue

        logs[key] /= logStep
        strLog = [key] + ["{:10.6f}".format(s) for s in logs[key]]
        print(formatCommand.format(*strLog))

    print('-'*50)


def saveLogs(data, pathLogs):
    with open(pathLogs, 'w') as file:
        json.dump(data, file, indent=2)


def getEncoder(encoderType, hiddenEncoder):

    if encoderType == 'mfcc':
        from model import MFCCEncoder
        return MFCCEncoder(hiddenEncoder)
    elif encoderType == 'lfb':
        from model import LFBEnconder
        return LFBEnconder(hiddenEncoder)
    else:
        from model import CPCEncoder
        return CPCEncoder(hiddenEncoder)


def getAR(args):
    if args.transformer:
        from transformers import buildTransformerAR
        arNet = buildTransformerAR(args.hiddenEncoder, 1,
                                   args.sizeWindow // 160, args.abspos)
        args.hiddenGar = args.hiddenEncoder
    else:
        from model import CPCAR
        arNet = CPCAR(args.hiddenEncoder, args.hiddenGar,
                      args.samplingType == "sequential",
                      args.nLevelsGRU)
    return arNet


def loadArgs(args, locArgs, forbiddenAttr=None):
    for k, v in locArgs.items():
        if forbiddenAttr is not None:
            if k not in forbiddenAttr:
                setattr(args, k, v)
        else:
            setattr(args, k, v)


def transferArgs(args, locArgs, toTransfer):
    for key in toTransfer:
        if key in locArgs:
            setattr(args, key, locArgs[key])


def getCheckpointData(pathDir):
    if not os.path.isdir(pathDir):
        return None
    checkpoints = [x for x in os.listdir(pathDir)
                   if os.path.splitext(x)[1] == '.pt']
    if len(checkpoints) == 0:
        print("No checkpoints found at " + pathDir)
        return None
    checkpoints.sort(key=lambda x: int(os.path.splitext(x[11:])[0]))
    data = os.path.join(pathDir, checkpoints[-1])
    with open(os.path.join(pathDir, 'checkpoint_logs.json'), 'rb') as file:
        logs = json.load(file)

    with open(os.path.join(pathDir, 'checkpoint_args.json'), 'rb') as file:
        args = json.load(file)

    return data, logs, args


def parseSeqLabels(pathLabels):
    lines = open(pathLabels, 'r').readlines()
    output = {"step": 160}  # Step in librispeech dataset is 160bits
    maxPhone = 0
    for line in lines:
        data = line.split()
        output[data[0]] = [int(x) for x in data[1:]]
        maxPhone = max(maxPhone, max(output[data[0]]))
    return output, maxPhone + 1


def cpuStats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())


def trainStep(model_criterion_combined,
              model_criterion,
              optimizer):

    if model_criterion.module.model.optimize:
        model_criterion.module.model.train()
    model_criterion.module.criterion.train()

    logs = {"step": 0}
    for step, fulldata in enumerate(model_criterion_combined):

        batchData, label = fulldata
        batchData = batchData.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        allLosses, allAcc = model_criterion(batchData, label)

        totLoss = allLosses.sum()
        totLoss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if "locLoss_train" not in logs:
            logs["locLoss_train"] = np.zeros(allLosses.size(1))
            logs["locAcc_train"] = np.zeros(allLosses.size(1))

        logs["step"] += 1
        logs["locLoss_train"] += (allLosses.mean(dim=0)).detach().cpu().numpy()
        logs["locAcc_train"] += (allAcc.mean(dim=0)).cpu().numpy()

    updateAndShowLogs("Update %d, training loss:" %
                      (logs["step"] + 1), logs, logs["locLoss_train"].shape[0])
    return logs


def valStep(dataLoader,
            model_criterion):
    model_criterion.eval()

    logs = {"step": 0}
    for step, fulldata in enumerate(dataLoader):

        batchData, label = fulldata

        batchData = batchData.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        allLosses, allAcc = model_criterion(batchData, label)

        if "locLoss_val" not in logs:
            logs["locLoss_val"] = np.zeros(allLosses.size(1))
            logs["locAcc_val"] = np.zeros(allLosses.size(1))

        logs["step"] += 1
        logs["locLoss_val"] += allLosses.mean(dim=0).cpu().numpy()
        logs["locAcc_val"] += allAcc.mean(dim=0).cpu().numpy()

    logs["step"] = step
    updateAndShowLogs("Validation loss:", logs, logs["locLoss_val"].shape[0])
    return logs


def run(trainLoader,
        valLoader,
        model_criterion,
        nEpoch,
        pathCheckpoint,
        optimizer,
        logs):

    print(f"Running {nEpoch} epochs")
    startEpoch = len(logs["epoch"])
    bestAcc = 0
    bestStateDict = None
    for epoch in range(startEpoch, nEpoch):

        print(f"Starting epoch {epoch}")
        print("Training dataset %d batches, Validation dataset %d batches" %
              (len(trainLoader), len(valLoader)))

        cpuStats()

        locLogsTrain = trainStep(
            trainLoader, model_criterion, optimizer)

        locLogsVal = valStep(valLoader, model_criterion)

        torch.cuda.empty_cache()

        currentAccuracy = float(locLogsVal["locAcc_val"].mean())
        if currentAccuracy > bestAcc:
            bestStateDict = model_criterion.module.model.state_dict()

        for key, value in dict(locLogsTrain, **locLogsVal).items():
            if key not in logs:
                logs[key] = [None for x in range(epoch)]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            logs[key].append(value)

        logs["epoch"].append(epoch)

        if pathCheckpoint is not None \
                and (epoch % 5 == 0 or epoch == nEpoch-1):
            print(pathCheckpoint)
            stateDict = {"gEncoder": model_criterion.module.model.state_dict(),
                         "cpcCriterion": model_criterion.module.criterion.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "best": bestStateDict}

            torch.save(stateDict, f"{pathCheckpoint}_{epoch}.pt")
            saveLogs(logs, pathCheckpoint + "_logs.json")


def main(args):
    args = parseArgs(args)
    print(f'CONFIG:\n{json.dumps(vars(args), indent=4, sort_keys=True)}')
    print('-' * 50)

    set_seed(args.random_seed)

    logs, loadOptimizer = {"epoch": []}, False
    if args.pathCheckpoint is not None and not args.restart:
        cdata = getCheckpointData(args.pathCheckpoint)
        if cdata is not None:
            data, logs, locArgs = cdata
            print(f"Checkpoint detected at {data}")
            loadArgs(args, locArgs,
                     forbiddenAttr={"nGPU", "pathCheckpoint",
                                    "debug", "restart"})
            args.load, loadOptimizer = [data], True

    seqNames, speakers = findAllSeqs(args.pathDB,
                                     recursionLevel=args.dataset_levels,
                                     extension=args.file_extension)

    # Datasets
    if args.pathTrain is not None:
        seqTrain = filterSeqs(args.pathTrain, seqNames)
    else:
        seqTrain = seqNames

    if args.pathVal is None:
        random.shuffle(seqTrain)
        sizeTrain = int(0.8 * len(seqTrain))
        seqTrain, seqVal = seqTrain[:sizeTrain], seqTrain[sizeTrain:]
    else:
        seqVal = filterSeqs(args.pathVal, seqNames)

    if args.debug:
        seqTrain = seqTrain[:2000]
        seqVal = seqVal[:2000]

    phoneLabels = None
    if args.supervised and args.pathPhone is not None:
        print("Loading the phone labels at " + args.pathPhone)
        phoneLabels, nPhones = parseSeqLabels(args.pathPhone)

    print(f'Loading audio data at {args.pathDB}')
    trainDataset = AudioBatchData(args.pathDB,
                                  args.sizeWindow,
                                  seqTrain,
                                  phoneLabels,
                                  list(speakers))

    valDataset = AudioBatchData(args.pathDB,
                                args.sizeWindow,
                                seqVal,
                                phoneLabels,
                                list(speakers))

    if args.load is not None:
        models = []
        hiddenGar, hiddenEncoder = 0, 0
        for path in args.load:
            print(f"Loading checkpoint {path}")
            _, _, locArgs = getCheckpointData(os.path.dirname(path))
            transferArgs(args, locArgs,
                        ["hiddenEncoder", "hiddenGar", "nLevelsGRU",
                         "transformer", "encoder_type", "reverse"])
            encoderNet = getEncoder(args.encoder_type, args.hiddenEncoder)
            arNet = getAR(args)
            state_dict = torch.load(path)
            m_ = CPCModel(encoderNet, arNet, args.reverse)
            m_.load_state_dict(state_dict["gEncoder"])
            models.append(m_)
            hiddenGar += locArgs["hiddenGar"]
            hiddenEncoder += locArgs["hiddenEncoder"]
        if len(models) == 1:
            cpcModel = models[0]
        else:
            if not args.eval:
                print(colored(f'WARNING: concatenated models not fit for \
                              training mode', 'red'))
            cpcModel = ConcatenatedModel(models)
        args.hiddenGar = hiddenGar
        args.hiddenEncoder = hiddenEncoder
    else:
        # Encoder network
        encoderNet = getEncoder(args.encoder_type, args.hiddenEncoder)

        # AR Network
        arNet = getAR(args)
        cpcModel = CPCModel(encoderNet, arNet, args.reverse)

    batchSize = args.nGPU * args.batchSizeGPU

    # Training criterion
    if not args.supervised:
        cpcCriterion = CPCUnsupersivedCriterion(args.nPredicts, args.hiddenGar,
                                                args.hiddenEncoder,
                                                args.negativeSamplingExt,
                                                args.reverse)
    elif args.pathPhone is not None:
        cpcCriterion = PhoneCriterion(args.hiddenGar, nPhones)
    else:
        cpcCriterion = SpeakerCriterion(args.hiddenGar,
                                        len(speakers))

    cpcModel.optimize = True
    if args.eval:
        print("Evaluation mode")
        cpcModel.optimize = False
        cpcModel.eval()
        for g in cpcModel.parameters():
            g.requires_grad = False

    # Optimizer
    g_params = list(cpcCriterion.parameters())

    if not args.eval:
        print("Optimizing model")
        g_params += list(cpcModel.parameters())

    optimizer = torch.optim.Adam(g_params, lr=args.learningRate,
                                 betas=(args.beta1, args.beta2),
                                 eps=args.epsilon)

    if loadOptimizer:
        print("Loading optimizer " + args.load[0])
        state_dict = torch.load(args.load[0])
        if "optimizer" in state_dict:
            optimizer.load_state_dict(state_dict["optimizer"])

    # Checkpoint
    if args.pathCheckpoint is not None:
        if not os.path.isdir(args.pathCheckpoint):
            os.mkdir(args.pathCheckpoint)
        args.pathCheckpoint = os.path.join(args.pathCheckpoint, "checkpoint")
        with open(args.pathCheckpoint + "_args.json", 'w') as file:
            json.dump(vars(args), file, indent=2)

    trainLoader = trainDataset.getDataLoader(batchSize, args.samplingType,
                                             not args.disable_offset,
                                             numWorkers=0)
    valLoader = valDataset.getDataLoader(batchSize, 'sequential', False,
                                         numWorkers=0)

    model_criterion = ModelCriterionCombined(cpcModel, cpcCriterion)
    model_criterion = torch.nn.DataParallel(model_criterion, device_ids=range(args.nGPU)).cuda()

    run(trainLoader,
        valLoader,
        model_criterion,
        args.nEpoch,
        args.pathCheckpoint,
        optimizer,
        logs)


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument(
        '--pathDB', type=str,
        default="/datasets01/LibriSpeech/022219/train-clean-100/")
    parser.add_argument('--pathTrain', type=str, default=None)
    parser.add_argument('--pathVal', type=str, default=None)
    parser.add_argument('--pathPhone', type=str, default=None)
    parser.add_argument('--hiddenEncoder', type=int, default=512)
    parser.add_argument('--hiddenGar', type=int, default=256)
    parser.add_argument('--nPredicts', type=int, default=12)
    parser.add_argument('--negativeSamplingExt', type=int, default=128)
    parser.add_argument('--supervised', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--load', type=str, default=None, nargs='*')
    parser.add_argument('--learningRate', type=float, default=2e-4)
    parser.add_argument('--schedulerStep', type=int, default=-1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-08)
    parser.add_argument('--pathCheckpoint', type=str, default=None)
    parser.add_argument('--sizeWindow', type=int, default=20480)
    parser.add_argument('--nEpoch', type=int, default=200)
    parser.add_argument('--samplingType', type=str, default='uniform',
                        choices=['samespeaker', 'uniform',
                                 'samesequence', 'sequential'])
    parser.add_argument('--nLevelsGRU', type=int, default=1)
    parser.add_argument('--nGPU', type=int, default=-1)
    parser.add_argument('--batchSizeGPU', type=int, default=8)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--file_extension', type=str, default=".flac")
    parser.add_argument('--dataset_levels', type=int, default=2)
    parser.add_argument('--disable_offset', action='store_true')
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--transformer', action='store_true')
    parser.add_argument('--abspos', action='store_true')
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--encoder_type', type=str,
                        choices=['cpc', 'mfcc', 'lfb'],
                        default='cpc')
    parser.add_argument('--random_seed', type=int, default=None)
    args = parser.parse_args(argv)

    # set it up if needed, so that it is dumped along with other args
    if args.random_seed is None:
        args.random_seed = random.randint(0, 2**31)

    if args.nGPU < 0:
        args.nGPU = torch.cuda.device_count()
    assert args.nGPU <= torch.cuda.device_count(), f"number of GPU asked: {args.nGPU}," \
        f"number GPU detected: {torch.cuda.device_count()}"
    print("Let's use", args.nGPU, "GPUs!")

    return args


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    main(args)
