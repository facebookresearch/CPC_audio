# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import sys
import torch
import json
from pathlib import Path
import ABX.abx_group_computation as abx_g
import ABX.abx_iterators as abx_it
from cpc.dataset import findAllSeqs
from cpc.feature_loader import buildFeature, FeatureModule, loadModel


def reduce_sparse_data(quotient, divisor):
    return quotient / (1e-08 * (divisor == 0) + divisor)


def ABX(feature_function,
        path_item_file,
        seq_list,
        distance_mode,
        step_feature,
        modes,
        cuda=False,
        max_x_across=5,
        max_size_group=30,
        normalize=True):

    # ABX dataset
    ABXDataset = abx_it.ABXFeatureLoader(path_item_file, seq_list,
                                         feature_function, step_feature, normalize)

    if cuda:
        ABXDataset.cuda()

    # Distance function
    distance_function = abx_g.get_distance_function_from_name(distance_mode)

    # Output
    scores = {}

    # ABX within
    if 'within' in modes:
        print("Computing ABX within speakers...")
        ABXIterator = ABXDataset.get_iterator('within', max_size_group)
        group_confusion = abx_g.get_abx_scores_dtw_on_group(ABXIterator,
                                                            distance_function,
                                                            ABXIterator.symmetric)
        n_data = group_confusion._values().size(0)
        index_ = torch.sparse.LongTensor(group_confusion._indices(),
                                         torch.ones((n_data),
                                                    dtype=torch.float),
                                         group_confusion.size())
        divisor_context = torch.sparse.sum(index_, dim=3).to_dense()
        group_confusion = torch.sparse.sum(group_confusion, dim=3).to_dense()
        group_confusion = reduce_sparse_data(group_confusion, divisor_context)
        S, p1, p2 = group_confusion.size()

        index_speaker = divisor_context > 0
        divisor_speaker = index_speaker.sum(dim=0)
        phone_confusion = reduce_sparse_data(group_confusion.sum(dim=0),
                                             divisor_speaker)

        scores['within'] = (phone_confusion.sum() /
                            (divisor_speaker > 0).sum()).item()
        print(f"...done. ABX within : {scores['within']}")

    # ABX across
    if 'across' in modes:
        print("Computing ABX across speakers...")
        ABXIterator = ABXDataset.get_iterator('across', max_size_group, max_x_across=max_x_across)

        group_confusion = abx_g.get_abx_scores_dtw_on_group(ABXIterator,
                                                            distance_function,
                                                            ABXIterator.symmetric)
        n_data = group_confusion._values().size(0)
        index_ = torch.sparse.LongTensor(group_confusion._indices(),
                                         torch.ones((n_data),
                                                    dtype=torch.float),
                                         group_confusion.size())

        divisor_context = torch.sparse.sum(index_, dim=[3]).to_dense()
        group_confusion = torch.sparse.sum(
             group_confusion, dim=[3]).to_dense()
        group_confusion = reduce_sparse_data(group_confusion, divisor_context)

        S1, p1, p2, S2 = group_confusion.size()
        index_speaker = divisor_context > 0
        divisor_speaker = index_speaker.sum(dim=0).sum(dim=2)
        phone_confusion = reduce_sparse_data(group_confusion.sum(dim=0).sum(dim=2),
                                             divisor_speaker)
        scores['across'] = (phone_confusion.sum() /
                             (divisor_speaker > 0).sum()).item()

        print(f"...done. ABX across : {scores['across']}")

    return scores


def update_base_parser(parser):
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--feature_size', type=float, default=0.01,
                        help="Size (in s) of one feature")
    parser.add_argument('--cuda', action='store_true',
                        help="Use the GPU to compute distances")
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'within', 'across'],
                        help="Type of ABX score to compute")
    parser.add_argument("--max_size_group", type=int, default=20,
                        help="Max size of a group while computing the"
                             "ABX score")
    parser.add_argument("--max_x_across", type=int, default=5,
                        help="When computing the ABX across score, maximum"
                             "number of speaker X to sample per couple A,B")
    parser.add_argument("--out", type=str, default=None,
                        help="Path where the results should be saved")
    parser.add_argument("--level_gru", type=int, default=None)


def parse_args(argv):

    base_parser = argparse.ArgumentParser(description='ABX metric')

    subparsers = base_parser.add_subparsers(dest='load')
    parser_checkpoint = subparsers.add_parser('from_checkpoint')
    update_base_parser(parser_checkpoint)
    parser_checkpoint.add_argument('path_checkpoint', type=str,
                                   help="Path to the model's checkpoint")
    parser_checkpoint.add_argument('path_item_file', type=str,
                                   help="Path to the ABX .item file containing "
                                   "the triplets labels")
    parser_checkpoint.add_argument('path_dataset', type=str,
                                   help="Path to the dataset")
    parser_checkpoint.add_argument('--seq_norm', action='store_true',
                                   help='If activated, normalize each batch '
                                   'of feature across the time channel before '
                                   'computing ABX.')
    parser_checkpoint.add_argument('--max_size_seq', default=64000, type=int,
                                   help='Maximal number of frames to consider '
                                   'when computing a batch of features.')
    parser_checkpoint.add_argument('--strict', action='store_true',
                                   help='If activated, each batch of feature '
                                   'will contain exactly max_size_seq frames.')
    parser_checkpoint.add_argument('--file_extension', type=str,
                                   default='.wav',
                                   help='Extension of ecah audio file in the '
                                   'dataset.')
    parser_checkpoint.add_argument('--get_encoded', action='store_true',
                                   help='If activated, compute the ABX score '
                                   'using the output of the encoder network.')
    parser_checkpoint.add_argument('-n', '--num_processes', type=int, default=40,
                                   help='Number of processes to use for group computation')

    parser_db = subparsers.add_parser('from_pre_computed')
    update_base_parser(parser_db)
    parser_db.add_argument('path_features', type=str,
                           help="Path to pre-computed torch features (.pt)")
    parser_db.add_argument('--file_extension', type=str,
                           default='.pt', help='Extension of each feature '
                           'in the dataset')

    # multi-gpu / multi-node
    return base_parser.parse_args(argv)


def main(argv):


    args = parse_args(argv)

    if args.load == 'from_checkpoint':
        updateConfig = None
        if args.level_gru is not None:
            updateConfig = argparse.Namespace(nLevelsGRU=args.level_gru)
        # Checkpoint
        model = loadModel([args.path_checkpoint], updateConfig=updateConfig)[0]
        model.gAR.keepHidden = True
        # Feature maker
        feature_maker = FeatureModule(model, args.get_encoded).cuda().eval()

        def feature_function(x): return buildFeature(feature_maker, x,
                                                     seqNorm=args.seq_norm,
                                                     strict=args.strict,
                                                     maxSizeSeq=args.max_size_seq)
    elif args.load == 'from_pre_computed':
        def feature_function(x): return torch.load(x, 'cpu')

    # Modes
    if args.mode == 'all':
        modes = ["within", "across"]
    else:
        modes = [args.mode]

    distance_mode = 'cosine'

    step_feature = 1 / args.feature_size

    # Get the list of sequences
    seq_list, _ = findAllSeqs(args.path_dataset, extension=args.file_extension)
    seq_list = [(str(Path(x).stem), str(Path(args.path_dataset) / x))
                for (_, x) in seq_list]

    if args.debug:
        seq_list = seq_list[:1000]

    scores = ABX(feature_function, args.path_item_file,
                 seq_list, distance_mode,
                 step_feature, modes,
                 cuda=args.cuda,
                 max_x_across=args.max_x_across,
                 max_size_group=args.max_size_group)

    out_dir = Path(args.path_checkpoint).parent if args.out is None \
        else Path(args.out)
    out_dir.mkdir(exist_ok=True)

    path_score = out_dir / 'ABX_scores.json'
    with open(path_score, 'w') as file:
        json.dump(scores, file, indent=2)

    path_args = out_dir / 'ABX_args.json'
    with open(path_args, 'w') as file:
        json.dump(vars(args), file, indent=2)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
