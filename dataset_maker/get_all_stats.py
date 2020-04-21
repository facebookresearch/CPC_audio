import argparse
from pathlib import Path
import sys
import numpy as np
from random import shuffle

import matplotlib.pyplot as plt
from cpc.dataset import findAllSeqs, filterSeqs
from utils.plot_stats import plot_hist, plot_as_hist
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

    return parser.parse_args(argv)

def main(args):

    path_cache = str(Path(args.output) / "_cache.txt")
    path_cache_speaker = str(Path(args.output) / "_cache_speaker.txt")
    if args.from_cache and Path(path_cache).is_file():
        sequence_data = sd.load_sequence_data(path_cache)
        speakers = sd.load_speaker_list(path_cache_speaker)
        n_speakers = len(speakers)
    else:
        sequences, speakers = findAllSeqs(args.path_db,
                                          extension=args.file_extension,
                                          loadCache=False)
        if args.debug:
            sequences = sequences[:100]
        if args.path_filter is not None:
            sequences = filterSeqs(args.path_filter, sequences)
        Path(args.output).mkdir(exist_ok=True)

        sd.save_speaker_list(speakers, path_cache_speaker)

        sequence_data = sd.get_sequence_data(args.path_db, sequences)
        n_speakers = len(speakers)

        # Save the metadata
        sd.save_sequence_data(sequence_data, path_cache)

    # Total size of the dataset
    tot_time = sum([x.time for x in sequence_data]) / 3600
    print(f"TOTAL SIZE {tot_time}h")


    # Get the speaker sizes
    speaker_sizes = sd.get_speakers_sizes(n_speakers, sequence_data) / 3600

    print(f"{len([x for x in speaker_sizes if x > 0])} speakers")

    fig, ax = plt.subplots(tight_layout=True)
    plot_as_hist(ax, speaker_sizes, Path(args.path_db).stem)
    ax.legend()
    fig.savefig(str(Path(args.output) / "speaker_stats.png"))

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

    #Get samples of amplitude 2
    amp2 = [x for x in sequence_data if x.max_amplitude >=1.99]

    #Get the size
    sizes_data = [x.time for x in sequence_data]
    fig, ax = plt.subplots(tight_layout=True)
    plot_hist(ax, sizes_data, Path(args.path_db).stem, 30)
    ax.legend()
    fig.savefig(str(Path(args.output) / "time_stats.png"))

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
