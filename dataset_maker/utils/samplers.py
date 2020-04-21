from random import sample, shuffle
from pathlib import Path


def estimate_balanced_speakers(speaker_sizes,
                               n_speakers,
                               target_size,
                               min_size_speaker):
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
    index_max = n_speakers
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


def filter_by_speaker_time_target(sequences, speaker_times):
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


def random_sampling(sequences, target_time):

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
