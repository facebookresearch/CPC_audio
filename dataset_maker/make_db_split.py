import argparse
from pathlib import Path
import sys
import utils.sequence_data as sd
import utils.samplers as samplers
import numpy as np
from random import shuffle, sample


def extract_subset(sequence_data,
                   speakers,
                   target_time,
                   n_speakers,
                   random_sampling=False,
                   gender_file=None):

    speaker_groups = []
    speaker_sizes = sd.get_speakers_sizes(len(speakers), sequence_data)
    if gender_file is not None:
        gender_index = sd.get_genders_from_file(speakers,
                                                gender_file)
        gender_sizes = sd.get_gender_sizes(speaker_sizes, gender_index)
        min_gender = np.argmin(gender_sizes)
        target_time_genders = [0, 0]
        target_time_genders[min_gender] = min(
            target_time / 2, gender_sizes[min_gender])
        target_time_genders[1 - min_gender] = target_time - \
            target_time_genders[min_gender]

        for _g in range(2):
            sizes_speakers_of_gender = np.zeros(n_speakers_original)
            for i, x in enumerate(gender_index):
                if x == sd.Gender(_g):
                    sizes_speakers_of_gender[i] = speaker_sizes[i]
            speaker_gender = min(
                n_speakers // 2, np.sum(sizes_speakers_of_gender > 0))
            speaker_groups.append(
                (sizes_speakers_of_gender, target_time_genders[_g], speaker_gender))

    else:
        speaker_groups.append((speaker_sizes, target_time, n_speakers))

    selected_sequences_train = []
    selected_sequences_remainer = []
    shuffle(sequence_data)
    index = 0
    for loc_speaker_size, loc_target_time, loc_target_n_speakers in speaker_groups:
        print(f"Gender {repr(sd.Gender(index))}")
        print(f"{loc_target_n_speakers} speakers detected")
        if random_sampling:
            loc_train_sequences, loc_remainers = samplers.random_sampling(sequence_data,
                                                                          loc_target_time)
        else:
            loc_speaker_selection = \
                samplers.estimate_balanced_speakers(loc_speaker_size,
                                                    loc_target_n_speakers,
                                                    loc_target_time,
                                                    args.min_time_speaker)
            loc_train_sequences, loc_remainers = samplers.filter_by_speaker_time_target(sequence_data,
                                                                                        loc_speaker_selection)
        selected_sequences_remainer += loc_remainers
        selected_sequences_train += loc_train_sequences
        index += 1

    return selected_sequences_train, selected_sequences_remainer


def parse_args(argv):

    parser = argparse.ArgumentParser(description='A tools which build a dataset '
                                     'split from its statistics')

    parser.add_argument('path_stats_dir',
                        help='Path to the directory containing the statistics '
                        'on the dataset')
    parser.add_argument('--file_extension', type=str, default=".wav")
    parser.add_argument('-o', '--output', type=str, default="coin")
    parser.add_argument('--n_speakers', type=int, default=251,
                        help='Number of speaker in the training subset')
    parser.add_argument('--gender_file', type=str, default=None,
                        help='Path to a file detailling the gender of the '
                        'speakers')
    parser.add_argument('--target_time', type=float, default=80,
                        help="Target size (in hours) of the training subset")
    parser.add_argument('--target_time_val', type=float, default=10,
                        help="Target size (in hours) of the validation subset")
    parser.add_argument('--target_time_dev', type=float, default=0,
                        help="Target size (in hours) of the dev subset")
    parser.add_argument('--target_time_test', type=float, default=0,
                        help="Target size (in hours) of the test subset")
    parser.add_argument('--min_time_speaker', type=float, default=0,
                        help="If > 0, remove all speakers with less than "
                        "min_time_speaker hours of data")
    parser.add_argument('--threshold_energy', type=float, default=0,
                        help="If > 0, remove all sequences with an energy"
                        "below the given threshold")
    parser.add_argument('--path_filter', type=str, default=None,
                        help="Filter the datset with the given list of "
                        "sequences before performing the split")
    parser.add_argument('--random_sampling', action='store_true',
                        help="If activated, do not try to get a balanced "
                        "speaker distribution")
    parser.add_argument('--ignore', type=str, default=None, nargs='*',
                        help='Exclude the given sequences from the '
                        'sampling')

    return parser.parse_args(argv)


def main(args):

    path_cache_sequence = str(Path(args.path_stats_dir) / "_cache.txt")
    path_cache_speaker = str(Path(args.path_stats_dir) / "_cache_speaker.txt")
    sequence_data = sd.load_sequence_data(path_cache_sequence)
    speakers = sd.load_speaker_list(path_cache_speaker)
    n_speakers_original = len(speakers)

    if args.path_filter is not None:
        sequence_data = sd.filter_sequence_data(
            sequence_data, args.path_filter)

    if args.ignore is not None:
        for file_name in args.ignore:
            print(f"Ignoring sequences from {file_name}")
            sequence_data = sd.remove_sequence_data(sequence_data, file_name)

    args.target_time *= 3600
    args.target_time_val *= 3600
    args.target_time_test *= 3600
    args.target_time_dev *= 3600
    if args.min_time_speaker is None:
        args.min_time_speaker = 0.8*args.target_time / args.n_speakers

    if args.threshold_energy is not None:
        sequence_data = [
            x for x in sequence_data if x.energy > args.threshold_energy]

    print(f"Saving the selection at {args.output}")

    if args.target_time_test > 0:
        print("Test set")
        selected_sequences_test, sequence_data =\
            extract_subset(sequence_data, speakers, args.target_time_test,
                           args.n_speakers, random_sampling=args.random_sampling,
                           gender_file=args.gender_file)
        path_test = Path(args.output).parent / \
            f'{Path(args.output).stem}_test.txt'
        sd.save_seq_list(selected_sequences_test, path_test)

    if args.target_time_dev > 0:
        print("Dev set")
        selected_sequences_dev, sequence_data =\
            extract_subset(sequence_data, speakers, args.target_time_dev,
                           args.n_speakers, random_sampling=args.random_sampling,
                           gender_file=args.gender_file)
        path_dev = Path(args.output).parent / \
            f'{Path(args.output).stem}_dev.txt'
        sd.save_seq_list(selected_sequences_dev, path_dev)

    if args.target_time > 0:
        print("Training set")
        selected_sequences_train, sequence_data =\
            extract_subset(sequence_data, speakers, args.target_time,
                           args.n_speakers, random_sampling=args.random_sampling,
                           gender_file=args.gender_file)
        path_train = Path(args.output).parent / \
            f'{Path(args.output).stem}_train.txt'
        sd.save_seq_list(selected_sequences_train, path_train)

    if args.target_time_val > 0:
        print("Validation set")
        selected_sequences_val, _ = extract_subset(sequence_data, speakers, args.target_time_val,
                                                   args.n_speakers, random_sampling=args.random_sampling,
                                                   gender_file=args.gender_file)
        path_val = Path(args.output).parent / \
            f'{Path(args.output).stem}_val.txt'
        sd.save_seq_list(selected_sequences_val, path_val)


if __name__ == "__main__":
    argv = sys.argv[1:]
    args = parse_args(argv)
    main(args)
