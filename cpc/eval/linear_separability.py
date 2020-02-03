# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import sys
import torch
import json
import time
import numpy as np
from pathlib import Path
from copy import deepcopy
import os

import cpc.criterion as cr
import cpc.feature_loader as fl
import cpc.utils.misc as utils
from cpc.dataset import AudioBatchData, findAllSeqs, filterSeqs, parseSeqLabels


def train_step(feature_maker, criterion, data_loader, optimizer):

    if feature_maker.optimize:
        feature_maker.train()
    criterion.train()

    logs = {"locLoss_train": 0,  "locAcc_train": 0}

    for step, fulldata in enumerate(data_loader):

        optimizer.zero_grad()
        batch_data, label = fulldata
        c_feature, encoded_data, _ = feature_maker(batch_data, None)
        if not feature_maker.optimize:
            c_feature, encoded_data = c_feature.detach(), encoded_data.detach()
        all_losses, all_acc = criterion(c_feature, encoded_data, label)
        totLoss = all_losses.sum()
        totLoss.backward()
        optimizer.step()

        logs["locLoss_train"] += np.asarray([all_losses.mean().item()])
        logs["locAcc_train"] += np.asarray([all_acc.mean().item()])

    logs = utils.update_logs(logs, step)
    logs["iter"] = step

    return logs


def val_step(feature_maker, criterion, data_loader):

    feature_maker.eval()
    criterion.eval()
    logs = {"locLoss_val": 0,  "locAcc_val": 0}

    for step, fulldata in enumerate(data_loader):

        with torch.no_grad():
            batch_data, label = fulldata
            c_feature, encoded_data, _ = feature_maker(batch_data, None)
            all_losses, all_acc = criterion(c_feature, encoded_data, label)

            logs["locLoss_val"] += np.asarray([all_losses.mean().item()])
            logs["locAcc_val"] += np.asarray([all_acc.mean().item()])

    logs = utils.update_logs(logs, step)

    return logs


def run(feature_maker,
        criterion,
        train_loader,
        val_loader,
        optimizer,
        logs,
        n_epochs,
        path_checkpoint):

    start_epoch = len(logs["epoch"])
    best_acc = -1

    start_time = time.time()

    for epoch in range(start_epoch, n_epochs):

        logs_train = train_step(feature_maker, criterion, train_loader,
                                optimizer)
        logs_val = val_step(feature_maker, criterion, val_loader)
        print('')
        print('_'*50)
        print(f'Ran {epoch + 1} epochs '
              f'in {time.time() - start_time:.2f} seconds')
        utils.show_logs("Training loss", logs_train)
        utils.show_logs("Validation loss", logs_val)
        print('_'*50)
        print('')

        if logs_val["locAcc_val"] > best_acc:
            best_state = deepcopy(fl.get_module(feature_maker).state_dict())
            best_acc = logs_val["locAcc_val"]

        logs["epoch"].append(epoch)
        for key, value in dict(logs_train, **logs_val).items():
            if key not in logs:
                logs[key] = [None for x in range(epoch)]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            logs[key].append(value)

        if (epoch % logs["saveStep"] == 0 and epoch > 0) or epoch == n_epochs - 1:
            model_state_dict = fl.get_module(feature_maker).state_dict()
            criterion_state_dict = fl.get_module(criterion).state_dict()

            fl.save_checkpoint(model_state_dict, criterion_state_dict,
                               optimizer.state_dict(), best_state,
                               f"{path_checkpoint}_{epoch}.pt")
            utils.save_logs(logs, f"{path_checkpoint}_logs.json")


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Linear separability trainer'
                                     ' (default test in speaker separability)')
    parser.add_argument('pathDB', type=str,
                        help="Path to the directory containing the audio data.")
    parser.add_argument('pathTrain', type=str,
                        help="Path to the list of the training sequences.")
    parser.add_argument('pathVal', type=str,
                        help="Path to the list of the test sequences.")
    parser.add_argument('load', type=str, nargs='*',
                        help="Path to the checkpoint to evaluate.")
    parser.add_argument('--pathPhone', type=str, default=None,
                        help="Path to the phone labels. If given, will"
                        " compute the phone separability.")
    parser.add_argument('--CTC', action='store_true',
                        help="Use the CTC loss (for phone separability only)")
    parser.add_argument('--pathCheckpoint', type=str, default='out',
                        help="Path of the output directory where the "
                        " checkpoints should be dumped.")
    parser.add_argument('--nGPU', type=int, default=-1,
                        help='Bumber of GPU. Default=-1, use all available '
                        'GPUs')
    parser.add_argument('--batchSizeGPU', type=int, default=8,
                        help='Batch size per GPU.')
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--debug', action='store_true',
                        help='If activated, will load only a small number '
                        'of audio data.')
    parser.add_argument('--unfrozen', action='store_true',
                        help="If activated, update the feature network as well"
                        " as the linear classifier")
    parser.add_argument('--no_pretraining', action='store_true',
                        help="If activated, work from an untrained model.")
    parser.add_argument('--file_extension', type=str, default=".flac",
                        help="Extension of the audio files in pathDB.")
    parser.add_argument('--save_step', type=int, default=-1,
                        help="Frequency at which a checkpoint should be saved,"
                        " et to -1 (default) to save only the best checkpoint.")
    parser.add_argument('--get_encoded', action='store_true',
                        help="If activated, will work with the output of the "
                        " convolutional encoder (see CPC's architecture).")
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Value of beta1 for the Adam optimizer.')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Value of beta2 for the Adam optimizer.')
    parser.add_argument('--epsilon', type=float, default=2e-8,
                        help='Value of epsilon for the Adam optimizer.')
    parser.add_argument('--ignore_cache', action='store_true',
                        help="Activate if the sequences in pathDB have"
                        " changed.")
    parser.add_argument('--size_window', type=int, default=20480,
                        help="Number of frames to consider in each batch.")
    args = parser.parse_args(argv)
    if args.nGPU < 0:
        args.nGPU = torch.cuda.device_count()
    if args.save_step <= 0:
        args.save_step = args.n_epoch

    args.load = [str(Path(x).resolve()) for x in args.load]
    args.pathCheckpoint = str(Path(args.pathCheckpoint).resolve())

    return args


def main(argv):

    args = parse_args(argv)
    logs = {"epoch": [], "iter": [], "saveStep": args.save_step}
    load_criterion = False

    seqNames, speakers = findAllSeqs(args.pathDB,
                                     extension=args.file_extension,
                                     loadCache=not args.ignore_cache)

    model, hidden_gar, hidden_encoder = fl.loadModel(args.load,
                                                     loadStateDict=not args.no_pretraining)
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(args.nGPU))

    dim_features = hidden_encoder if args.get_encoded else hidden_gar

    # Now the criterion
    phone_labels = None
    if args.pathPhone is not None:
        phone_labels, n_phones = parseSeqLabels(args.pathPhone)
        if not args.CTC:
            print(f"Running phone separability with aligned phones")
            criterion = cr.PhoneCriterion(dim_features,
                                          n_phones, args.get_encoded)
        else:
            print(f"Running phone separability with CTC loss")
            criterion = cr.CTCPhoneCriterion(dim_features,
                                             n_phones, args.get_encoded)
    else:
        print(f"Running speaker separability")
        criterion = cr.SpeakerCriterion(dim_features, len(speakers))
    criterion.cuda()
    criterion = torch.nn.DataParallel(criterion, device_ids=range(args.nGPU))

    # Dataset
    seq_train = filterSeqs(args.pathTrain, seqNames)
    seq_val = filterSeqs(args.pathVal, seqNames)

    if args.debug:
        seq_train = seq_train[:1000]
        seq_val = seq_val[:100]

    db_train = AudioBatchData(args.pathDB, args.size_window, seq_train,
                              phone_labels, len(speakers))
    db_val = AudioBatchData(args.pathDB, args.size_window, seq_val,
                            phone_labels, len(speakers))

    batch_size = args.batchSizeGPU * args.nGPU

    train_loader = db_train.getDataLoader(batch_size, "uniform", True,
                                          numWorkers=0)

    val_loader = db_val.getDataLoader(batch_size, 'sequential', False,
                                      numWorkers=0)

    # Optimizer
    g_params = list(criterion.parameters())
    model.optimize = False
    model.eval()
    if args.unfrozen:
        print("Working in full fine-tune mode")
        g_params += list(model.parameters())
        model.optimize = True
    else:
        print("Working with frozen features")
        for g in model.parameters():
            g.requires_grad = False

    optimizer = torch.optim.Adam(g_params, lr=args.lr,
                                 betas=(args.beta1, args.beta2),
                                 eps=args.epsilon)

    # Checkpoint directory
    args.pathCheckpoint = Path(args.pathCheckpoint)
    args.pathCheckpoint.mkdir(exist_ok=True)
    args.pathCheckpoint = str(args.pathCheckpoint / "checkpoint")

    with open(f"{args.pathCheckpoint}_args.json", 'w') as file:
        json.dump(vars(args), file, indent=2)

    run(model, criterion, train_loader, val_loader, optimizer, logs,
        args.n_epoch, args.pathCheckpoint)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    args = sys.argv[1:]
    main(args)
