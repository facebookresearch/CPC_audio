from progressbar import ProgressBar
from pathlib import Path
import torchaudio
import numpy as np
import tqdm
from enum import IntEnum
from typing import Dict, NamedTuple, Tuple, List, Union, Set, Callable
from multiprocessing import Pool
import tqdm


class SequenceData(NamedTuple):
    r"""
    A NamedTuple summing up the metadata of a sequence in the dataset
    """
    name: str                  # File's name
    speaker: int               # Speaker's index
    time: float                # Size (in seconds) of the audio files
    max_amplitude: float       # sequence.max() - sequence.min()
    energy: float              # Standard deviation of the audio sequence
    perplexity: float = None  # If given, the average perplexity of a
    # classifier on the audio sequence

    def get_updated_value(self, **kwargs):
        r"""
        NamedTuples are not mutable : this function allows it
        """
        new_kwargs = {name: kwargs.get(name, getattr(self, name))
                      for name in self._fields}
        return SequenceData(**new_kwargs)


class Gender(IntEnum):
    FEMALE = 0
    MALE = 1


class AudioMetaLoader:
    r"""
    A class warpping the get_sequence_data_from_file in order to load
    a series of data using the multiprocessing library.
    """

    def __init__(self, path_db):
        r"""
        Initialize the loader with the root of the dataset
        """
        self.path_db = Path(path_db)

    def get_sequence_data_from_file(self, seq_data: Tuple[int, str]) -> SequenceData:
        r"""
        Compute the sequence data object correspinding to the given tuple
        speaker, seq_name where:
        - speaker (int) is the index / id of the speaker
        - seq_name (str) is the relative path of the sequence in the dataset
        """
        speaker, seq_name = seq_data
        full_path = str(self.path_db / seq_name)
        try:
            frame_data, sr = torchaudio.load(full_path)
            l = frame_data.size(1) / sr
            out = SequenceData(name=str(Path(full_path).stem),
                               speaker=speaker,
                               time=l,
                               max_amplitude=frame_data.max().item() - frame_data.min().item(),
                               energy=frame_data.mean(dim=0).std().item())
            del frame_data
            return out
        except RuntimeError:
            return None


def get_sequence_data(path_db,
                      file_and_speakers_list: List[Tuple[int, str]],
                      n_process: int = 64) -> List[SequenceData]:
    r"""
    Load the sequence data corresponding to a list of Tuples
    speaker, sequence_name (see AudioMetaLoader.get_sequence_data_from_file
    above)
    """

    file_reader = AudioMetaLoader(path_db)
    out = []

    with Pool(n_process) as p:
        for x in tqdm.tqdm(p.imap_unordered(file_reader.get_sequence_data_from_file,
                                            file_and_speakers_list,
                                            chunksize=10), total=len(file_and_speakers_list)):
            if x is not None:
                out.append(x)
    return out


def load_field_from_tsv(path_tsv_file: Path,
                        key: str,
                        type_: Callable):
    r"""
    Load the data relative to a specific field from a tsv file
    - path_tsv_file : Path to the file to read
    - key : name of the field to extract
    - type_ : type of the value to extract (float, int, str etc..)

    returns Dict[str, type_]
    """

    with open(path_perplexity, 'r') as file:
        perplexity_data = [x.strip() for x in file.readlines()]

    order_perplexity = get_order_from_header(perplexity_data[0])
    index_name = order_perplexity["name"]
    index_key = order_perplexity[key]
    out = {}

    for line in perplexity_data[1:]:
        data = line.split()
        name = data[index_name]
        out[name] = type_(data[index_key])
    return out


def update_sequence_data_with_val(sequence_data: List[SequenceData],
                                  val_dict: Dict[str, Union[str, float]],
                                  key: str) -> List[SequenceData]:

    out = []
    for seq in sequence_data:
        if seq.name in val_dict:
            out.append(seq.get_updated_value(**{key: val_dict[seq.name]}))
    return out


def filter_by_speaker(sequence_data: List[SequenceData],
                      speaker_list: Union[List[int], Set[int]]) -> List[SequenceData]:

    out = []
    for data in sequence_data:
        if data.speaker in speaker_list:
            out.append(data)
    return out


def filter_sequence_data(sequence_data: List[SequenceData],
                         path_filter: Path) -> List[SequenceData]:

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


def remove_sequence_data(sequence_data: List[SequenceData],
                         path_filter: Path) -> List[SequenceData]:

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


def save_sequence_data(sequence_data: List[SequenceData],
                       path_out: Path) -> None:

    with open(path_out, 'w') as file:
        file.write("# name speaker time max_amplitude energy perplexity\n")
        for seq in sequence_data:
            file.write(f"{seq.name} ")
            file.write(f"{seq.speaker} ")
            file.write(f"{seq.time} ")
            file.write(f"{seq.max_amplitude} ")
            file.write(f"{seq.energy} ")
            file.write(f"{seq.perplexity}\n")


def get_order_from_header(header: str):
    assert(header[0] == '#')
    header = header[1:].strip()
    return {x: i for i, x in enumerate(header.split())}


def load_sequence_data(path_in: Path):

    out = []
    with open(path_in, 'r') as file:
        lines = file.readlines()

    order = get_order_from_header(lines[0])

    def get_val(data, key, mandatory, type_):
        if mandatory:
            return type_(data[order[key]])
        i_ = order.get(key, None)
        if i_ is None or data[i_] == 'None':
            return None
        return type_(data[i_])

    for line in lines[1:]:
        data = line.split()
        out.append(SequenceData(name=get_val(data, "name", True, str),
                                speaker=get_val(data, "speaker", True, int),
                                time=get_val(data, "time", True, float),
                                max_amplitude=get_val(
                                    data, "max_amplitude", True, float),
                                energy=get_val(
                                    data, "max_amplitude", False, float),
                                perplexity=get_val(data, "perplexity", False, float)))
    return out


def save_seq_list(sequence_data: List[SequenceData], path_out: Path) -> None:
    r"""
    Only save the names of the sequences
    """
    with open(path_out, 'w') as file:
        for seq in sequence_data:
            file.write(seq.name + '\n')


def save_speaker_list(speaker_list: List[int], path_out: Path) -> None:
    with open(path_out, 'w') as file:
        for speaker in speaker_list:
            file.write(speaker + '\n')


def load_speaker_list(path_in: Path):
    with open(path_in, 'r') as file:
        lines = file.readlines()

    return [x.strip() for x in lines]


def correct_speaker_index(sequence_data: List[SequenceData]) -> Tuple[List[SequenceData], Dict[int, int]]:
    r"""
    Rename the speakers in order to get indexes ranging from 0 to n_speakers -1
    """

    speakers = {}
    n_speakers = 0
    for index, sequence in enumerate(sequence_data):
        if sequence.speaker not in speakers:
            speakers[sequence.speaker] = n_speakers
            n_speakers += 1
        sequence_data[index] = sequence._replace(
            speaker=speakers[sequence.speaker])

    return sequence_data, n_speakers


def get_speakers_sizes(n_speakers: int,
                       sequence_data_list: List[SequenceData]):

    out_speakers = np.zeros(n_speakers)
    for data in sequence_data_list:
        out_speakers[data.speaker] += data.time
    return out_speakers


def get_gender_sizes(speaker_sizes: List[int],
                     gender_index: List[Gender]):
    out = np.zeros(2)
    for index, size in enumerate(speaker_sizes):
        out[int(gender_index[index])] += size
    return out


def get_genders_from_file(speaker_list: List[int],
                          gender_file: Path) -> List[Gender]:
    with open(gender_file, 'r') as file:
        header = file.readline()
        is_magica = is_magica_header(header)

    if is_magica:
        return get_speaker_gender_magicdata(speaker_list, gender_file)
    else:
        return get_speaker_genders_aishell(speaker_list, gender_file)


def get_speaker_genders_aishell(speaker_list: List[int], gender_file: Path) -> List[Gender]:
    r"""
    Load the genders from the AISHELL dataset
    """
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


def is_magica_header(line: str) -> bool:

    data = line.strip().split()
    out = len(data) == 4
    if not out:
        return out
    out = out and data[0] == 'SPKID'
    out = out and data[1] == 'Age'
    out = out and data[2] == 'Gender'
    out = out and data[3] == 'Dialect'

    return out


def get_speaker_gender_magicdata(speaker_list: List[int], gender_file: Path) -> List[Gender]:
    r"""
    Load the genders from the MAGICDATA dataset
    """
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
