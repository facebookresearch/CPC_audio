from os.path import join, exists, dirname, splitext, basename, abspath
import argparse
import json
import os
from pathlib import Path
import torch
from cpc.feature_loader import buildFeature, FeatureModule, loadModel
from cpc.clustering.clustering import loadClusterModule
import cpc.eval.ABX.abx_iterators as abx_it
import cpc.eval.ABX.abx_group_computation as abx_g
from cpc.eval.eval_ABX import ABX
from time import time


def write_json(filepath, scores):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as file:
        json.dump(scores, file, indent=2)


def read_args(pathArgs):
    print(f"Loading args from {pathArgs}")
    with open(pathArgs, "r") as file:
        args = argparse.Namespace(**json.load(file))

    return args


def load_cpc_feature_maker(
    CPC_path_checkpoint, encoder_layer=False, keepHidden=True, gru_level=-1
):
    updateConfig = None
    if gru_level is not None:
        updateConfig = argparse.Namespace(nLevelsGRU=gru_level)
    model, hiddenGar, _ = loadModel(
        [CPC_path_checkpoint], loadStateDict=True, updateConfig=updateConfig
    )
    model.gAR.keepHidden = keepHidden
    feature_maker = FeatureModule(model, get_encoded=encoder_layer)
    feature_maker.eval()
    feature_maker.cuda()
    print("Checkpoint loaded!")
    print("")

    return feature_maker


class ClusteringFeatures:
    def __init__(
        self,
        clustering_path_checkpoint,
        soft_clustering=False,
        encoder_layer=False,
        keepHidden=True,
        group_modes="concat",
        onehot_dict=None,
    ):

        self.group_modes = group_modes
        self.soft_clustering = soft_clustering

        # Load Clustering args
        clustering_path_checkpoint = Path(clustering_path_checkpoint)
        assert clustering_path_checkpoint.suffix == ".pt"
        assert self.group_modes in ["seq", "onehot", "concat", "combine"]
        if (clustering_path_checkpoint.parent / "args.json").is_file():
            path_config = clustering_path_checkpoint.parent / "args.json"
        elif (clustering_path_checkpoint.parent / "checkpoint_args.json").is_file():
            path_config = clustering_path_checkpoint.parent / "checkpoint_args.json"
        else:
            raise RuntimeError(
                f"Args file not found in the directory {clustering_path_checkpoint.parent}"
            )
        clustering_args = read_args(path_config)
        print("")
        print(
            f"Clutering args:\n{json.dumps(vars(clustering_args), indent=4, sort_keys=True)}"
        )
        print("-" * 50)

        # Define feature_function
        ## Load CPC featureMaker
        level_gru = vars(clustering_args).get("level_gru", None)
        self.featureMaker = load_cpc_feature_maker(
            clustering_args.pathCheckpoint,
            encoder_layer=encoder_layer,
            keepHidden=keepHidden,
            gru_level=level_gru,
        )
        n_features = self.featureMaker.out_feature_dim
        ### Load Clustering module
        self.clusterModule = loadClusterModule(clustering_path_checkpoint)
        print("Checkpoint loaded!")
        print("")

        self.dim_clusters = self.clusterModule.Ck.shape[-1]
        self.n_groups = n_features // self.dim_clusters
        assert (
            n_features % self.dim_clusters == 0
        ), f"Number of features {n_features} must be divided by the dimension of clusters {self.dim_clusters}"

        if self.n_groups > 1 and self.group_modes == "onehot":
            assert (
                onehot_dict is not None
            ), "A dictionary must be given when there are more than one group and in onehot mode!"
            with open(onehot_dict, "r") as f:
                lines = f.read().split("\n")
            pair2idx = {word.split()[0]: i for i, word in enumerate(lines) if word}

    def feature_function(self, x):
        c_feature = buildFeature(
            self.featureMaker, x, seqNorm=False, strict=True, maxSizeSeq=64000
        ).cuda()
        c_feature = c_feature.view(1, -1, self.dim_clusters)
        dist_clusters = self.clusterModule(c_feature)
        if self.soft_clustering:
            return dist_clusters[0]
        else:
            q_feature = torch.argmin(dist_clusters, dim=-1)
            if self.n_groups > 1:
                if self.group_modes == "seq":
                    n_clusters = self.clusterModule.Ck.shape[1]
                    one_hot_feature = torch.FloatTensor(
                        q_feature.shape[-1], n_clusters
                    ).cuda()
                    one_hot_feature.zero_()
                    one_hot_feature.scatter_(-1, q_feature[0].unsqueeze(1), 1)
                    one_hot_feature = one_hot_feature.view(-1, n_clusters)

                elif self.group_modes == "onehot":
                    q_feature = q_feature[0].detach().cpu().numpy()
                    q_feature = [
                        pair2idx[pair]
                        for pair in [
                            "-".join([str(i) for i in item])
                            for item in q_feature.reshape(-1, 2)
                        ]
                    ]
                    q_feature = torch.tensor(q_feature).unsqueeze(0).cuda()
                    n_clusters = len(pair2idx)
                    one_hot_feature = torch.FloatTensor(
                        q_feature.shape[-1], n_clusters
                    ).cuda()
                    one_hot_feature.zero_()
                    one_hot_feature.scatter_(-1, q_feature[0].unsqueeze(1), 1)
                    print(
                        one_hot_feature.size(),
                    )

                elif self.group_modes == "concat":
                    n_clusters = self.clusterModule.Ck.shape[1] * self.n_groups
                    one_hot_feature = torch.FloatTensor(
                        q_feature.shape[-1], n_clusters // self.n_groups
                    ).cuda()
                    one_hot_feature.zero_()
                    one_hot_feature.scatter_(-1, q_feature[0].unsqueeze(1), 1)
                    one_hot_feature = one_hot_feature.view(-1, n_clusters)

                elif self.group_modes == "combine":
                    n_clusters = self.clusterModule.Ck.shape[1]
                    one_hot_feature = torch.FloatTensor(
                        q_feature.shape[-1] // self.n_groups, n_clusters
                    ).cuda()
                    one_hot_feature.zero_()
                    one_hot_feature.scatter_(-1, q_feature[0][::2].unsqueeze(1), 1)
                    one_hot_feature.scatter_(-1, q_feature[0][1::2].unsqueeze(1), 1)

            else:
                n_clusters = self.clusterModule.Ck.shape[1]
                one_hot_feature = torch.FloatTensor(
                    q_feature.shape[-1], n_clusters
                ).cuda()
                one_hot_feature.zero_()
                one_hot_feature.scatter_(-1, q_feature[0].unsqueeze(1), 1)

            S, N = one_hot_feature.size()
            return one_hot_feature.view(1, S, N)

    @property
    def step_feature_multiplication(self):

        if self.group_modes == "seq":
            return self.n_groups

        return 1


class QuantizedClustering:
    def __init__(self, quantized_file, onehot_dict=None):

        # Define feature_function
        self.frames_dict = {}
        with open(quantized_file, "r") as f:
            for line in f:
                if line[-1] == "\n":
                    line = line[:-1]
                filename, frames = line.split("\t")
                filename = splitext(basename(filename))[0]
                self.frames_dict[filename] = frames

        frames = next(iter(self.frames_dict.values()))
        if not frames.split(",")[0].isdigit():  # multi-group ie. 65-241
            assert (
                onehot_dict is not None
            ), "A dictionary must be given when the quantized outputs is not digits (multi-group case)!"

        if onehot_dict:
            print("")
            print(f"Loading onehot dictionary from {onehot_dict}...")
            with open(onehot_dict, "r") as f:
                lines = f.read().split("\n")
            pair2idx = {word.split()[0]: i for i, word in enumerate(lines) if word}

        self.n_units = -1  # Number of quantized units
        for filename in self.frames_dict:
            frames = self.frames_dict[filename].split(",")
            if onehot_dict:
                idxs_seq = [pair2idx[item] for item in frames]  # sequence of idxs
            else:
                idxs_seq = [int(item) for item in frames]
            self.n_units = max(self.n_units, max(idxs_seq))
            self.frames_dict[filename] = idxs_seq
        self.n_units += 1  # idxs start from 0
        print("")
        print(f"Number of quantized units: {self.n_units}")

    def feature_function(self, x):

        filename = splitext(basename(x))[0]
        idxs_seq = torch.tensor(self.frames_dict[filename])
        one_hot_feature = torch.FloatTensor(len(idxs_seq), self.n_units)
        one_hot_feature.zero_()
        one_hot_feature.scatter_(-1, idxs_seq.unsqueeze(1), 1)
        N, S = one_hot_feature.size()

        return one_hot_feature.view(1, N, S).cuda()

    @property
    def step_feature_multiplication(self):
        return 1


def eval_ABX_Librispeech(
    path_data,
    path_item_file,
    feature_function,
    modes="within",
    feature_size=0.01,
    distance_mode="cosine",
    file_extension=".flac",
    debug=False,
    path_output=None,
):

    save = path_output is not None
    # Some assertions
    assert modes in ["within", "across", "all"]
    assert distance_mode in ["cosine", "euclidian"]
    if save:
        assert not exists(
            path_output
        ), f"The output file {path_output} already exists!!"

    # Modes
    if modes in ["within", "across"]:
        modes = [modes]
    elif modes == "all":
        modes = ["within", "across"]

    # Get step_feature
    step_feature = 1 / feature_size

    # Get the list of sequences
    seq_list = list(Path(path_data).glob(f"**/*{file_extension}"))
    if debug:
        seq_list = seq_list[:100]

    seq_list = [(x.stem, str(x)) for x in seq_list]

    # Do ABX
    scores = ABX(
        feature_function,
        path_item_file,
        seq_list,
        distance_mode,
        step_feature,
        modes,
        cuda=False,
        max_x_across=5,
        max_size_group=10,
        normalize=True,
    )

    # save
    if save:
        scores["args"] = {}
        scores["args"]["modes"] = modes
        scores["args"]["feature_size"] = feature_size
        scores["args"]["distance_mode"] = distance_mode
        scores["args"]["path_data"] = str(path_data)
        scores["args"]["file_extension"] = file_extension
        scores["args"]["debug"] = debug
        if debug:
            scores["args"]["debug_size"] = len(seq_list)
        write_json(path_output, scores)

    return scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ABX Evaluation on CPC-clustering. Note that use either --clustering pathClustering, --CPC pathCPC, or --quantized pathQuantized"
    )
    group_type_input = parser.add_mutually_exclusive_group(required=True)
    group_type_input.add_argument(
        "--quantized",
        type=str,
        help="(str) The path of the quantized file of the corresponding eval dataset.",
        default=None,
    )
    group_type_input.add_argument(
        "--clustering",
        type=str,
        help="(str) The checkpoint of the clustering module.",
        default=None,
    )
    parser.add_argument(
        "--name-output",
        type=str,
        help="(str) The name of the output in the scores directory.",
        default=None,
    )
    parser.add_argument(
        "--modes",
        choices=["all", "within", "across"],
        help="Mode of the ABX evaluation. Default: all.",
        default="all",
    )
    parser.add_argument(
        "--feature-size",
        type=float,
        help="1/sample_rate. Default: 0.01 (~100 Hz)",
        default=0.01,
    )
    parser.add_argument(
        "--gru",
        type=int,
        help="Level of GRU to be taken when loading CPC module. Default: -1 (last level)",
        default=-1,
    )
    parser.add_argument(
        "--file-extension",
        type=str,
        default=".flac"
    )
    parser.add_argument(
        "--soft-clustering",
        "-s",
        action="store_true",
        help="Whether to use soft clustering (distances to clusters) features.",
    )
    parser.add_argument(
        "--group-modes",
        choices=["seq", "onehot", "concat", "combine"],
        help="Method to deal with multiple groups. Default: onehot.",
        default="onehot",
    )
    parser.add_argument(
        "--onehot-dict",
        type=str,
        help="Path to the dictionary of the quantized units (required when there are more than one group and in onehot mode).",
        default=None,
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save the results."
    )
    parser.add_argument(
        "--path_audio_data",
        type=str,
        help="(str) The path to the Librispeech dataset.",
        default="/datasets01/librispeech/062419/dev-clean",
    )
    parser.add_argument(
        "--path_abx_item",
        type=str,
        help="(str) The path to the ABX data file, can be found in"
        "https://dl.fbaipublicfiles.com/librilight/data/ABX_data.tgz",
        default="/checkpoint/mriviere/ABX_librilight/dev-clean.item",
    )
    args = parser.parse_args()

    if args.clustering:
        FeatureMaker = ClusteringFeatures(
            args.clustering,
            soft_clustering=args.soft_clustering,
            encoder_layer=False,
            keepHidden=True,
            group_modes=args.group_modes,
            onehot_dict=args.onehot_dict,
        )
    elif args.quantized:
        assert (
            args.group_modes == "onehot",
            "Only one-hot grouping is available when working with quantized features !",
        )
        FeatureMaker = QuantizedClustering(args.quantized, onehot_dict=args.onehot_dict)

    print("")
    print(f"Feature function args:\n{json.dumps(vars(args), indent=4, sort_keys=True)}")
    print("-" * 50)

    step_feature_multiplication = FeatureMaker.step_feature_multiplication
    if step_feature_multiplication > 1:
        feature_size = args.feature_size / step_feature_multiplication
    else:
        feature_size = args.feature_size

    scores = eval_ABX_Librispeech(
        path_data=args.path_audio_data,
        path_item_file=args.path_abx_item,
        feature_function=FeatureMaker.feature_function,
        modes=args.modes,
        feature_size=feature_size,
        distance_mode="cosine",
        file_extension=args.file_extension,
        debug=args.debug,
        path_output=args.name_output,
    )
