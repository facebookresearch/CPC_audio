import argparse
from pathlib import Path
import sys
import numpy as np
from random import shuffle
import math

import matplotlib.pyplot as plt
from cpc.dataset import findAllSeqs, filterSeqs
from utils.plot_stats import plot_hist, plot_as_hist, plot_edge_hist
from utils.math_utils import get_entropy
import utils.sequence_data as sd


def parse_args(argv):

    parser = argparse.ArgumentParser(description='Dataset statistics, '
                                     'you need to run this script before '
                                     'using make_db_split')

    parser.add_argument('path_db', type=str,
                        help="Path to the dataset")
    parser.add_argument('--file_extension', type=str, default=".wav")
    parser.add_argument('-o', '--output', type=str, default="coin")
    parser.add_argument('--from_cache', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gender_file', type=str, default=None,
                        help="Path to a file detailling the gender of the "
                        "different speakers")
    parser.add_argument('--path_filter', type=str, default=None,
                        help="Build statistics only on the given subset")
    parser.add_argument('--path_perplexity', type=str, default=None,
                        help="Build statistics only on the given subset")
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--world_size', type=int, default=-1)
    parser.add_argument('--load_cache', type=str, default=None)

    return parser.parse_args(argv)


def remove_path(seq_list, path_db):
    if path_db[-1] != '/':
        path_db += "/"
    s = len(path_db)
    return [(x[0], x[1][s:]) for x in path_db]


def main(args):

    path_cache = str(Path(args.output) / "_cache.txt")
    path_cache_speaker = str(Path(args.output) / "_cache_speaker.txt")
    if args.local_rank >= 0:
        path_cache = str(Path(args.output) /
                         f"_cache_{args.local_rank}_{args.world_size}.txt")
        path_cache_speaker = str(
            Path(args.output) / f"_cache_speaker_{args.local_rank}_{args.world_size}.txt")
    if args.from_cache and Path(path_cache).is_file():
        sequence_data = sd.load_sequence_data(path_cache)
        speakers = sd.load_speaker_list(path_cache_speaker)
        n_speakers = len(speakers)
    elif args.load_cache is not None:
        sequence_data = sd.load_sequence_data(
            Path(args.load_cache) / "_cache.txt")
        speakers = sd.load_speaker_list(Path(args.load_cache) / "_cache.txt")
        if args.path_filter is not None:
            sequence_data = sd.filter_sequence_data(
                sequence_data, args.path_filter)
        n_speakers = len(speakers)
    else:
        Path(args.output).mkdir(exist_ok=True)
        sequences, speakers = findAllSeqs(args.path_db,
                                          extension=args.file_extension,
                                          loadCache=False)
        if args.path_filter is not None:
            sequences = filterSeqs(args.path_filter, sequences)

        if args.debug:
            shuffle(sequences)
            sequences = sequences[:10000]

        if args.local_rank >= 0:
            start = (len(sequences) * args.local_rank) // args.world_size
            end = (len(sequences) * (args.local_rank+1)) // args.world_size
            sequences = sequences[start:end]

        sd.save_speaker_list(speakers, path_cache_speaker)

        sequence_data = sd.get_sequence_data(args.path_db, sequences)
        n_speakers = len(speakers)

    # Total size of the dataset
    tot_time = sum([x.time for x in sequence_data]) / 3600
    print(f"TOTAL SIZE {tot_time}h")

    # Get the speaker sizes
    speaker_sizes = sd.get_speakers_sizes(n_speakers, sequence_data) / 3600
    n_active_speakers = len([x for x in speaker_sizes if x > 0])

    print(f"{n_active_speakers} speakers")

    fig, ax = plt.subplots(tight_layout=True)
    plot_as_hist(ax, speaker_sizes, Path(args.path_db).stem)
    ax.legend()
    fig.savefig(str(Path(args.output) / "speaker_stats.png"))

    speaker_entropy = get_entropy(speaker_sizes)
    print(f"Speaker entropy : {speaker_entropy}")
    print(f"Normalized entropy [(speaker entropy) / log(N_speakers)] :"
        f" {speaker_entropy / math.log(n_active_speakers)}")

    # Get the genders
    if args.gender_file is not None:
        gender_index = sd.get_genders_from_file(speakers,
                                                args.gender_file)
        gender_sizes = sd.get_gender_sizes(speaker_sizes, gender_index)
        fig, ax = plt.subplots(tight_layout=True)
        ax.bar([0], [gender_sizes[0]], label="female")
        ax.bar([1], [gender_sizes[1]], label="male")
        ax.legend()
        fig.savefig(str(Path(args.output) / "gender_stats.png"))

    # Get the energy
    energy_data = np.array([x.energy for x in sequence_data])
    fig, ax = plt.subplots(tight_layout=True)
    plot_hist(ax, energy_data, Path(args.path_db).stem, 30)
    ax.legend()
    fig.savefig(str(Path(args.output) / "energy_stats.png"))

    # Get the amplitude
    amplitude_data = np.array([x.max_amplitude for x in sequence_data])
    fig, ax = plt.subplots(tight_layout=True)
    plot_hist(ax, amplitude_data, Path(args.path_db).stem, 30)
    ax.legend()
    fig.savefig(str(Path(args.output) / "amplitude_stats.png"))

    # Get samples of amplitude 2
    amp2 = [x for x in sequence_data if x.max_amplitude >= 1.99]

    # Get the size
    sizes_data = [x.time for x in sequence_data]
    fig, ax = plt.subplots(tight_layout=True)
    plot_hist(ax, sizes_data, Path(args.path_db).stem, 30)
    ax.legend()
    fig.savefig(str(Path(args.output) / "time_stats.png"))

    # Get the perplexity
    if args.path_perplexity is not None:
        perplexity_dict = sd.load_field_from_tsv(args.path_perplexity,
                                                 "mean_perplexity",
                                                 float)
        sequence_data = sd.update_sequence_data_with_val(sequence_data,
                                                         perplexity_dict,
                                                         "perplexity")
        perplexity_data = [math.log(x.perplexity) for x in sequence_data]
        fig, ax = plt.subplots(tight_layout=True)
        plot_edge_hist(ax, perplexity_data, Path(args.path_db).stem, 100)
        ax.legend()
        fig.savefig(str(Path(args.output) / "perplexity_stats.png"))

        avg_perplexity = sum(
            [x.perplexity for x in sequence_data]) / len(sequence_data)
        print(f"Average perplexity : {avg_perplexity}")

    # Save the metadata
    if not args.from_cache:
        sd.save_sequence_data(sequence_data, path_cache)

    print(f"Min size: {min(sizes_data)}")
    print(f"Max size: {max(sizes_data)}")
    sizes_data.sort()

    n_data = len(sizes_data)

    print(f"D1_10 {sizes_data[n_data // 10]}")
    print(f"Q1_4 {sizes_data[n_data // 4]}")
    print(f"median {sizes_data[n_data // 2]}")
    print(f"avg {sum(sizes_data)/n_data}")


if __name__ == "__main__":

    argv = sys.argv[1:]
    args = parse_args(argv)
    main(args)
