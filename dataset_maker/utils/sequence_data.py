from progressbar import ProgressBar
from typing import List, NamedTuple
from pathlib import Path
import torchaudio
import numpy as np
from enum import IntEnum


class SequenceData(NamedTuple):
    name: str
    speaker: int
    time: float
    max_amplitude: float
    energy: float


class Gender(IntEnum):
    FEMALE = 0
    MALE = 1


def get_sequence_data(path_db, file_and_speakers_list):

    bar = ProgressBar(maxval=len(file_and_speakers_list))
    path_db = Path(path_db)
    out = []

    for index, seq_data in enumerate(file_and_speakers_list):

        speaker, seq_name = seq_data
        bar.update(index)
        full_path = str(path_db / seq_name)
        try:
            frame_data, sr = torchaudio.load(full_path)
        except RuntimeError:
            continue
        l = frame_data.size(1) / sr

        out.append(SequenceData(name=str(Path(seq_name).stem),
                                speaker=speaker,
                                time=l,
                                max_amplitude=frame_data.max() - frame_data.min(),
                                energy=frame_data.mean(dim=0).std()))
    bar.finish()
    return out


def filter_by_speaker(sequence_data, speaker_list):

    out = []
    for data in sequence_data:
        if data.speaker in speaker_list:
            out.append(data)
    return out


def filter_sequence_data(sequence_data, path_filter):

    with open(path_filter, 'r') as file:
        name_filter = [x.strip() for x in file.readlines()]

    sequence_data.sort(key=lambda x: x.name)
    name_filter.sort()

    out = []
    index_filter = 0
    for data in sequence_data:

        while index_filter < len(name_filter) and name_filter[index_filter] < data.name:
            index_filter += 1

        if name_filter[index_filter] == data.name:
            out.append(data)

    return out


def remove_sequence_data(sequence_data, path_filter):

    with open(path_filter, 'r') as file:
        name_filter = [x.strip() for x in file.readlines()]

    sequence_data.sort(key=lambda x: x.name)
    name_filter.sort()

    out = []
    index_filter = 0
    for data in sequence_data:

        while index_filter < len(name_filter) and name_filter[index_filter] < data.name:
            index_filter += 1

        if index_filter >= len(name_filter) or name_filter[index_filter] > data.name:
            out.append(data)

    return out


def save_sequence_data(sequence_data, path_out):

    with open(path_out, 'w') as file:
        file.write("# name speaker time max_amplitude energy")
        for seq in sequence_data:
            file.write(f"{seq.name} ")
            file.write(f"{seq.speaker} ")
            file.write(f"{seq.time} ")
            file.write(f"{seq.max_amplitude} ")
            file.write(f"{seq.energy}\n")


def load_sequence_data(path_in):

    out = []
    with open(path_in, 'r') as file:
        lines = file.readlines()[1:]

    for line in lines:
        data = line.split()
        out.append(SequenceData(name=data[0],
                                speaker=int(data[1]),
                                time=float(data[2]),
                                max_amplitude=float(data[3]),
                                energy=float(data[4])))
    return out


def save_seq_list(sequence_data, path_out):

    with open(path_out, 'w') as file:
        for seq in sequence_data:
            file.write(seq.name + '\n')


def save_speaker_list(speaker_list, path_out):
    with open(path_out, 'w') as file:
        for speaker in speaker_list:
            file.write(speaker + '\n')


def load_speaker_list(path_in):
    with open(path_in, 'r') as file:
        lines = file.readlines()

    return [x.strip() for x in lines]


def correct_speaker_index(sequence_data):

    speakers = {}
    n_speakers = 0
    for index, sequence in enumerate(sequence_data):
        if sequence.speaker not in speakers:
            speakers[sequence.speaker] = n_speakers
            n_speakers += 1
        sequence_data[index] = sequence._replace(
            speaker=speakers[sequence.speaker])

    return sequence_data, n_speakers


def get_speakers_sizes(n_speakers, sequence_data_list):

    out_speakers = np.zeros(n_speakers)
    for data in sequence_data_list:
        out_speakers[data.speaker] += data.time
    return out_speakers


def get_gender_sizes(speaker_sizes, gender_index):
    out = np.zeros(2)
    for index, size in enumerate(speaker_sizes):
        out[int(gender_index[index])] += size
    return out


def get_genders_from_file(speaker_list, gender_file):
    with open(gender_file, 'r') as file:
        header = file.readline()
        is_magica = is_magica_header(header)

    if is_magica:
        return get_speaker_gender_magicdata(speaker_list, gender_file)
    else:
        return get_speaker_genders_aishell(speaker_list, gender_file)


def get_speaker_genders_aishell(speaker_list, gender_file):

    speaker_dict = {x: i for i, x in enumerate(speaker_list)}
    out_genders = [None for x in speaker_list]
    with open(gender_file, 'r') as file:
        data = file.readlines()

    for line in data:
        name, str_gender = line.strip().split()
        name = f'S{name}'
        gender = Gender.FEMALE if str_gender == 'F' else Gender.MALE
        if name in speaker_dict:
            out_genders[speaker_dict[name]] = gender

    speakers_not_found = [speaker_list[i]
                          for x, i in enumerate(out_genders) if x is None]
    if len(speakers_not_found) > 0:
        raise RuntimeError(f"Gender not found for speakers {speaker_list[i]}")

    return out_genders


def is_magica_header(line):

    data = line.strip().split()
    out = len(data) == 4
    if not out:
        return out
    out = out and data[0] == 'SPKID'
    out = out and data[1] == 'Age'
    out = out and data[2] == 'Gender'
    out = out and data[3] == 'Dialect'

    return out


def get_speaker_gender_magicdata(speaker_list, gender_file):

    speaker_dict = {x: i for i, x in enumerate(speaker_list)}
    out_genders = [None for x in speaker_list]

    with open(gender_file, 'r') as file:
        data = file.readlines()

    gender_dict = {'female': Gender.FEMALE, 'male': Gender.MALE}
    assert(is_magica_header(data[0]))

    for line in data[1:]:
        id, age, gender_str = line.strip().split()[:3]
        gender = gender_dict[gender_str]
        if id in speaker_dict:
            out_genders[speaker_dict[id]] = gender

    speakers_not_found = [speaker_list[i]
                          for x, i in enumerate(out_genders) if x is None]
    if len(speakers_not_found) > 0:
        raise RuntimeError(f"Gender not found for speakers {speaker_list[i]}")

    return out_genders
