# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import torchaudio
import progressbar
import os
import sys
from pathlib import Path


def adjust_sample_rate(path_db, file_list, path_db_out,
                       target_sr):
    bar = progressbar.ProgressBar(maxval=len(file_list))
    bar.start()

    for index, item in enumerate(file_list):
        path_in = os.path.join(path_db, item)
        path_out = os.path.join(path_db_out, item)

        bar.update(index)
        data, sr = torchaudio.load(path_in)
        transform = torchaudio.transforms.Resample(orig_freq=sr,
                                                   new_freq=target_sr,
                                                   resampling_method='sinc_interpolation')
        data = transform(data)
        torchaudio.save(path_out, data, target_sr,
                        precision=16, channels_first=True)

    bar.finish()


def get_names_list(path_tsv_file):

    with open(path_tsv_file, 'r') as file:
        data = file.readlines()

    return [x.split()[0] for x in data]


def parse_args(argv):

    parser = argparse.ArgumentParser(description='Adjust the sample rate of '
                                     'a given group of audio files')

    parser.add_argument('path_db', type=str,
                        help='Path to the directory containing the audio '
                        'files')
    parser.add_argument("path_phone_files", type=str,
                        help='Path to the .txt file containing the list of '
                        'the files with a phone transcription')
    parser.add_argument("path_out", type=str,
                        help='Path out the output directory')
    parser.add_argument("--out_sample_rate", type=int, default=16000,
                        help="Sample rate of the output audio files "
                        "(default is 160000)")
    parser.add_argument('--file_extension', type=str, default='.mp3')

    return parser.parse_args(argv)


def main(argv):

    args = parse_args(argv)

    file_list_db = [f for f in os.listdir(args.path_db)
                    if Path(f).suffix == args.file_extension]

    print(f"Found {len(file_list_db)} in the dataset")
    file_list_phone = get_names_list(args.path_phone_files)
    print(f"Found {len(file_list_phone)} with a phone transcription")

    file_list_db.sort()
    file_list_phone.sort()
    out_list = []
    index_phone = 0
    for file_name in file_list_db:
        while Path(file_name).stem > file_list_phone[index_phone]:
            index_phone += 1
            if index_phone >= len(file_list_phone):
                break
        if Path(file_name).stem == file_list_phone[index_phone]:
            out_list.append(file_name)

    print(f"Converting {len(out_list)} files")

    Path(args.path_out).mkdir(parents=True, exist_ok=True)
    adjust_sample_rate(args.path_db, out_list,
                       args.path_out, args.out_sample_rate)


if __name__ == '__main__':
    main(sys.argv[1:])
