from random import sample, shuffle
from pathlib import Path
from typing import Dict, List
from .sequence_data import SequenceData


def get_common_seqs(ref_1: List[str], ref_2: List[str]) -> List[str]:
    ref_1.sort()
    ref_2.sort()

    i_ = 0
    out = []
    for seq in ref_1:
        while i_ < len(ref_2) and Path(ref_2[i_]).stem < Path(seq).stem:
            i_ += 1

        if i_ < len(ref_2) and Path(ref_2[i_]).stem == Path(seq).stem:
            out.append(seq)

    return out


def get_top_by_attr(sequence_data: List[SequenceData],
                    key: str,
                    target_time: float,
                    reverse=False) -> List[SequenceData]:
    sequence_data.sort(key=lambda x: getattr(x, key), reverse=reverse)
    tot_time = 0
    out = []
    for seq in sequence_data:
        out.append(seq)
        tot_time += seq.time
        if tot_time >= target_time:
            break
    return out


def get_top_n_speakers(speaker_size: List[float], n_speakers: int):
    sorted_speaker_indexed = [(x, i) for i, x in enumerate(speaker_sizes)]
    sorted_speaker_indexed.sort(reverse=True)
    return [x[1] for x in sorted_speaker_indexed[:n_speakers]]


def estimate_balanced_speakers(speaker_sizes: List[float],
                               n_speakers: int,
                               target_size: float,
                               min_size_speaker: float) -> Dict[int, float]:
    r"""
    Estimate the quantity of data to extract for each speaker in order
    to get a speaker distribution as bakanced as possible containing n_speakers
    """
    sorted_speaker_indexed = [(x, i) for i, x in enumerate(speaker_sizes)
                              if x > min_size_speaker]
    sorted_speaker_indexed.sort(reverse=True)
    sorted_speaker_indexed = sorted_speaker_indexed[:n_speakers]

    working_data = [x[0] for x in sorted_speaker_indexed]

    curr_time = 0
    index_max = len(sorted_speaker_indexed)
    output = {x[1]: 0 for x in sorted_speaker_indexed}

    while curr_time < target_size:
        next_target_size = (target_size - curr_time) / n_speakers
        if working_data[index_max - 1] >= next_target_size:
            index_cut = index_max
        else:
            index_cut = next(x[0] for x in enumerate(
                working_data[:index_max]) if x[1] < next_target_size)
        for i in range(index_cut):
            working_data[i] -= next_target_size
            output[sorted_speaker_indexed[i][1]] += next_target_size
            curr_time += next_target_size
        for j in range(index_cut, index_max):
            output[sorted_speaker_indexed[j][1]] += working_data[j]
            curr_time += working_data[j]
            working_data[j] = 0

        if index_cut == index_max:
            break
        if index_cut == 0:
            break
        index_max = index_cut
    return output


def filter_by_speaker_time_target(sequences: List[SequenceData],
                                  speaker_times: Dict[int, float]) -> List[SequenceData]:
    r"""
    Samples randomly sequences from the input list so that the speaker
    distribution matched the one described by speaker_times
    """
    shuffle(sequences)
    out = []
    remainers = []
    curr_times = {x: 0 for x in speaker_times}
    for seq in sequences:
        if seq.speaker in speaker_times:
            if curr_times[seq.speaker] + seq.time < speaker_times[seq.speaker]:
                out.append(seq)
                curr_times[seq.speaker] += seq.time
            else:
                remainers.append(seq)
        else:
            remainers.append(seq)
    tot_time = sum([x for k, x in curr_times.items()])
    print(f"{tot_time / 3600} hours sampled")
    return out, remainers


def random_sampling(sequences: List[SequenceData], target_time: float):

    shuffle(sequences)
    t = 0
    index = 0
    for seq in sequences:
        index += 1
        t += seq.time
        if t > target_time:
            break

    print(f"{t / 3600} hours sampled")
    return sequences[:index], sequences[index:]
