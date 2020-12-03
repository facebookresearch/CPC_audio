# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import csv
import os
import argparse
import json
import torchaudio
import string
import progressbar
import shutil
from pathlib import Path
from random import shuffle, choice
from phonem_parser import getPhoneTranscription
from cpc.dataset import findAllSeqs
from utils.samplers import get_common_seqs
import tqdm
from multiprocessing import Pool


def spotKSampleWithPhonem(phoneDict, phonemIndex, k=-1):

    out = []
    if k < 0:
        k = len(phoneDict)
    for sentence in phoneDict:
        for phone in phoneDict[sentence]:
            if phone == phonemIndex:
                out.append(sentence)
                break
        if len(out) > k:
            break

    return out


def to_cpc_org(speakerData, path_clips_in, file_extension):

    speakers_index = {}
    ns = 0
    path_clips_in = Path(path_clips_in)
    bar = progressbar.ProgressBar(maxval=len(speakerData))
    i = 0
    bar.start()
    for seq_name, speaker_id in speakerData.items():

        bar.update(i)
        i += 1

        if speaker_id not in speakers_index:
            speakers_index[speaker_id] = str(ns)
            ns += 1

        path_dir_speaker = path_clips_in / speakers_index[speaker_id]
        path_seq_in = (path_clips_in / seq_name).with_suffix(file_extension)
        if path_seq_in.is_file():
            path_dir_speaker.mkdir(exist_ok=True)
            path_seq_out = path_dir_speaker / path_seq_in.name
            shutil.move(path_seq_in, path_seq_out)

    bar.finish()


def getSpeakerDataFromTSV(pathTSV):

    out = {}
    with open(pathTSV,  'r', encoding="utf-8") as tsvfile:
        reader = csv.DictReader(tsvfile, dialect='excel-tab')
        for row in reader:
            id = row['client_id']
            name = row['path']
            out[name] = id
    return out


def getSpeakerTimeStats(pathClips, speakerData):
    out = {}
    for seqName, speakerName in speakerData.items():
        path = os.path.join(pathClips, seqName)
        if not os.path.isfile(path):
            continue
        if speakerName not in out:
            out[speakerName] = 0
        info = torchaudio.info(path)[0]
        # Lenght in second of the sequence
        out[speakerName] += info.length / (info.rate * 3600)
    return out


def saveSeqList(path, seqList):
    with open(path, 'w') as file:
        for item in seqList:
            file.write(item + '\n')


def loadSeqList(path):

    with open(path, 'r') as file:
        data = file.readlines()

    return [x.replace('\n', '') for x in data]


def replace_sil(phoneTranscriptions, char_sil, end_sil=False):
    out = []
    for x in phoneTranscriptions:
        y = x.strip().split()
        if end_sil:
            y += ['']
        out.append(f"-{char_sil}-".join(y))
    return out


def strip_sil(phoneTranscriptions, char_sil):
    out = []
    for x in phoneTranscriptions:
        y = x.strip().split()
        out.append("-".join(y) + f"-{char_sil}")
    return out


def getPhoneConverter(phoneTranscriptions, converter, stats):
    for item in phoneTranscriptions:
        phones = item.replace(' ', '-').split('-')
        phones = [x for x in phones if x != '']
        for val in phones:
            if val not in converter:
                size = len(converter)
                converter[val] = size
            if val not in stats:
                stats[val] = 0
            stats[val] += 1
    return converter, stats


def applyConverter(phoneTranscriptions, converter):

    output = []
    for item in phoneTranscriptions:
        splitted = item.replace(' ', '-').split('-')
        sentence = ' '.join([str(converter[x]) for x in splitted if x != ''])
        output.append(sentence)
    return output


def lookForPhonem(phoneTranscriptions, phoneIndex):
    with open(phoneTranscriptions, 'r') as file:
        data = file.readlines()

    for line in data:
        line = line.replace('\n', '')
        items = line.split()
        phones = [int(x) for x in items[1:]]
        if phoneIndex in phones:
            print(items[0])


def writeData(pathList, transcriptions, pathOutput):
    print(len(pathList), len(transcriptions))
    assert(len(pathList) == len(transcriptions))
    nItems = len(pathList)
    with open(pathOutput, 'w') as file:
        for i in range(nItems):
            file.write(
                f"{os.path.splitext(pathList[i])[0]} {transcriptions[i]} \n")


def loadPhoneDict(pathTranscription):
    out = {}
    with open(pathTranscription, 'r') as file:
        data = file.readlines()
    for line in data:
        line = line.replace('\n', '')
        items = line.split()
        out[os.path.splitext(items[0])[0]] = [int(x) for x in items[1:]]
    return out


def savePhoneDict(phoneDict, pathOut):
    with open(pathOut, 'w') as file:
        for key, value in phoneDict.items():
            file.write(f"{key} {' '.join([str(x) for x in value])} \n")


def getPhonesList(phoneDict):
    out = set()
    for item, value in phoneDict.items():
        out = out.union(set(value))
    return out


def getNewPhoneConverter(phoneList, phoneConverter):
    #phoneList = getPhonesList(phoneDict)
    print(f"{len(phoneList)} phones detected")

    newPhoneConverter = {}
    backwardConverter = {}

    reverseOldConverter = {val: key for key, val in phoneConverter.items()}
    for index, phone in enumerate(list(phoneList)):
        char_conv = reverseOldConverter[phone]
        newPhoneConverter[char_conv] = index
        backwardConverter[phone] = index
    return newPhoneConverter, backwardConverter


def applyBackwardConverter(phoneDict, backwardConverter, strict=False):
    output = {}
    for item, value in phoneDict.items():
        output[item] = [backwardConverter[x] for x in value]
    return output


def removeSeqsNotWithTranscription(phoneDict, backwardConverter):
    output = {}
    for seqName, seqPhones in phoneDict.items():
        take = True
        for x in seqPhones:
            if x not in backwardConverter:
                take = False
                break
        if take:
            output[seqName] = seqPhones
    return output


def removeSeqsWithPhone(seqList, invalidPhones, phoneDict):

    out = []
    for seq in seqList:
        base = os.path.splitext(seq)[0]
        take = True
        if base not in phoneDict:
            continue
        for phone in invalidPhones:
            if phone in phoneDict[base]:
                take = False
                break
        if take:
            out.append(seq)
    return out


def tsvToPhoneData(pathCsV, code, pathOutput, converter, stats, removeNonEnglishLetters=False):
    textList = []
    pathList = []
    punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    punc += string.punctuation
    punc += "¿¡»«°ºʿ½€−‑"
    charac = {}
    punc += (b'\xef\xbf\xbd').decode('utf-8')
    letters = "abcdefghijklmnopqrstuvwxyz "
    with open(pathCsV,  'r', encoding="utf-8") as tsvfile:
        reader = csv.DictReader(tsvfile, dialect='excel-tab')
        for row in reader:
            sentence = row['sentence']
            for char in punc:
                sentence = sentence.replace(char, ' ')
            take = len(sentence.replace(' ', '')) > 0
            if removeNonEnglishLetters:
                s = sentence.lower()
                for item in s:
                    if item not in letters:
                        take = False
                        break
            if not take:
                continue
            for item in sentence:
                if item not in charac:
                    charac[item] = 0
                charac[item] += 1
            textList.append(sentence)
            pathList.append(row['path'])
    phoneTranscriptions = getPhoneTranscription(textList, code)
    converter, stats = getPhoneConverter(phoneTranscriptions, converter, stats)
    intPhones = applyConverter(phoneTranscriptions, converter)
    writeData(pathList, intPhones, pathOutput)
    return converter, stats


def removeBigSeqs(pathDB, pathList, maxTime):

    output = []
    bar = progressbar.ProgressBar(len(pathList))
    bar.start()
    for index, item in enumerate(pathList):
        bar.update(index)
        data, sampleRate = torchaudio.load(os.path.join(pathDB, item))
        if data.size(1) // sampleRate < maxTime:
            output.append(item)

    bar.finish()

    return output


def extractSeqListFromTSV(pathTSV):
    pathList = []
    with open(pathTSV) as tsvfile:
        reader = csv.DictReader(tsvfile, dialect='excel-tab')
        for row in reader:
            pathList.append(Path(row['path']).stem)
    return pathList


def extractPhoneFromTxtPhones(pathPhones, ext=".mp3"):
    with open(pathPhones, 'r') as phoneFile:
        data = phoneFile.readlines()

    return [l.split()[0] + ext for l in data]


def makeTrainValSplits(pathClips,
                       trainList,
                       targetTimeTrain,
                       targetTimeVal,
                       strict=False):

    shuffle(trainList)
    lossTrain = 1
    lossVal = 1
    outTrain = []
    outVal = []

    sizeTrain, sizeVal = 0, 0
    for index, seqName in enumerate(trainList):
        path_seq = os.path.join(pathClips, seqName)
        if not os.path.isfile(path_seq):
            continue
        info = torchaudio.info(path_seq)[0]
        # Lenght in second of the sequence
        l_ = info.length / (info.rate * 3600)

        seqName = os.path.splitext(seqName)[0]

        if lossTrain >= lossVal and lossTrain > 0:
            outTrain.append(seqName)
            sizeTrain += l_
            lossTrain = (targetTimeTrain - sizeTrain) / targetTimeTrain
        elif lossVal > lossTrain and lossVal > 0:
            outVal.append(seqName)
            sizeVal += l_
            lossVal = (targetTimeVal - sizeVal) / targetTimeVal
        else:
            break

    if strict and sizeTrain < 0.9*targetTimeTrain:
        print("Not enough training data")
        print(f"Aimed at {targetTimeTrain}, but retrieved {sizeTrain}")
        return None, None

    print(f"Training dataset: {sizeTrain } hours")
    print(f"Validation dataset: {sizeVal } hours")
    return outTrain, outVal


def extractTimeSplit(pathClips, clipList, targetTime):
    shuffle(clipList)
    out = []
    currTime, last_index = 0, 0
    for seqName in clipList:
        path = os.path.join(pathClips, seqName)
        if not os.path.isfile(path):
            print(path)
            continue
        info = torchaudio.info(path)[0]
        # Lenght in second of the sequence
        l_ = info.length / (info.rate * 3600)
        currTime += l_
        out.append(os.path.splitext(seqName)[0])
        last_index += 1

        if currTime > targetTime:
            break

    print(f"Extracted {currTime} hours")
    return out, clipList[last_index:]


def extractTimeSplitSpeakerUniform(pathClips, clipList, speakerData, targetTime):

    speakerSampler = {}
    for clip in clipList:
        speaker = speakerData[clip]
        if speaker not in speakerSampler:
            speakerSampler[speaker] = []
        speakerSampler[speaker].append(clip)

    speakerKeys = list(speakerSampler.keys())

    out = []
    currTime = 0
    nItems = len(clipList)
    for i in range(nItems):
        iSpeaker = choice(range(len(speakerKeys)))
        speaker = speakerKeys[iSpeaker]
        seqName = speakerSampler[speaker][-1]
        path = os.path.join(pathClips, seqName)
        if not os.path.isfile(path):
            del speakerSampler[speaker][-1]
            if len(speakerSampler[speaker]) == 0:
                del speakerKeys[iSpeaker]
            continue
        info = torchaudio.info(path)[0]
        # Lenght in second of the sequence
        l_ = info.length / (info.rate * 3600)
        currTime += l_
        out.append(os.path.splitext(seqName)[0])

        if currTime > targetTime:
            break

        del speakerSampler[speaker][-1]
        if len(speakerSampler[speaker]) == 0:
            del speakerKeys[iSpeaker]

    print(f"Target time {targetTime}, retrieved {currTime}")

    return out


def makeSplitSpeakerWise(pathClips, speakerData, speakerStats, t1, t2):

    fullTime = sum([x for _, x in speakerStats.items()])
    time1, time2 = 0, 0
    speakers1, speakers2 = [], []
    loss1, loss2 = 1, 1
    share1 = t1 / (t1 + t2)
    targetTime1 = fullTime * share1
    targetTime2 = fullTime - targetTime1

    shuffledData = [(x, k) for k, x in speakerStats.items()]
    shuffledData.sort()

    for sizeSpeaker, speaker in shuffledData:
        if loss1 >= loss2 and loss1 > 0:
            time1 += sizeSpeaker
            loss1 = ((targetTime1 - time1) / targetTime1)
            speakers1.append(speaker)
        elif loss2 > 0:
            time2 += sizeSpeaker
            loss2 = ((targetTime2 - time2) / targetTime2)
            speakers2.append(speaker)
        else:
            break

    out1, out2 = [], []
    for seqName, speaker in speakerData.items():
        if speaker in speakers1:
            out1.append(seqName)
        elif speaker in speakers2:
            out2.append(seqName)

    print(f"Group one {len(speakers1)} speakers")
    print(f"Group two {len(speakers2)} speakers")

    out1 = extractTimeSplitSpeakerUniform(pathClips, out1, speakerData, t1)
    out2 = extractTimeSplitSpeakerUniform(pathClips, out2, speakerData, t2)

    return out1, out2


def adjust_sr(data):
    pathIn, pathOut, new_sr = data
    if os.path.isfile(pathOut) and not force:
        return
    if not os.path.isfile(pathIn):
        return
    data, sr = torchaudio.load(pathIn)
    sampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=new_sr,
                                             resampling_method='sinc_interpolation')
    data = sampler(data)
    torchaudio.save(pathOut, data, new_sr)


def adjustSampleRate(pathDB, pathList, pathDBOut,
                     force=False, new_sr=16000,
                     n_process=10):

    to_deal = []
    for item in pathList:
        pathIn = os.path.join(pathDB, item)
        pathOut = os.path.join(pathDBOut, item)
        to_deal.append((pathIn, pathOut, new_sr))

    with Pool(n_process) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(adjust_sr, to_deal), total=len(to_deal)):
            pass


def applyGain(pathDB, pathDBOut, pathList, gain):
    bar = progressbar.ProgressBar(len(pathList))
    bar.start()
    for index, item in enumerate(pathList):
        bar.update(index)
        pathIn = os.path.join(pathDB, item)
        pathOut = os.path.join(pathDBOut, item)
        data, sr = torchaudio.load(pathIn)
        data *= gain
        torchaudio.save(pathOut, data, sr)
    bar.finish()


def make_split(list_item, target_time, uniform_split, name_dir_clip="clips_16k"):
    if uniform_split:
        selected, remainers = extractTimeSplit(os.path.join(args.pathDB,
                                                            name_dir_clip),
                                               list_item, args.time_test)
    else:
        speakerData = {x: globalSpeakerData[x] for x in list_item}
        speakerStats = getSpeakerTimeStats(os.path.join(args.pathDB,
                                                        name_dir_clip),
                                           speakerData)
        totTimeDB = sum([x for _, x in speakerStats.items()])
        selected, remainers = makeSplitSpeakerWise(os.path.join(args.pathDB, name_dir_clip),
                                                   speakerData, speakerStats, target_time,
                                                   totTimeDB-targetTimeTest)
    return selected, remainers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Common voices dataset preparation')
    subparsers = parser.add_subparsers(dest='command')

    parser_prepare = subparsers.add_parser('prepare_phone')
    parser_prepare.add_argument('pathDB', type=str)
    parser_prepare.add_argument('languageCode', type=str)
    parser_prepare.add_argument('--target', type=str, default=None,
                                choices=['dev', 'test', 'validated',
                                         'invalidated'])

    parser_invalid = subparsers.add_parser('remove_phone')
    parser_invalid.add_argument('pathDB', type=str)
    parser_invalid.add_argument('-i', '--invalidPhones', type=str, nargs='*',
                                default=None)
    parser_invalid.add_argument('-m', '--min_occ', type=int, default=-1)
    parser_invalid.add_argument('--phoneLimit', type=int, default=-1)

    parser_train = subparsers.add_parser('make')
    parser_train.add_argument('pathDB', type=str)
    parser_train.add_argument('--to_16k', action='store_true')
    parser_train.add_argument('--to_cpc', action='store_true')
    parser_train.add_argument('-g', '--gain', type=float, default=1.)
    parser_train.add_argument('--spot', type=str, default=None)

    parser_split = subparsers.add_parser('get_filter')
    parser_split.add_argument('pathDB', type=str)

    parser_lookup = subparsers.add_parser('lookup')
    parser_lookup.add_argument('phone_transcript', type=str)
    parser_lookup.add_argument('index', type=int)

    parser_show = subparsers.add_parser('show_phones')
    parser_show.add_argument('pathDB', type=str)
    parser_show.add_argument('--reduced', action='store_true')
    args = parser.parse_args()

    # Convert the transcriptions to phone data
    if args.command == 'prepare_phone':
        targets = ["dev.tsv", "test.tsv", "train.tsv", "validated.tsv"]
        if args.target is not None:
            targets = [args.target]
        converter, stats = {}, {}
        for targetFile in targets:
            print(f"Transcribing {targetFile}")
            pathIn = os.path.join(args.pathDB, targetFile)
            pathOut = os.path.join(args.pathDB,
                                   f"{os.path.splitext(targetFile)[0]}_phones.txt")
            converter, stats = tsvToPhoneData(pathIn, args.languageCode,
                                              pathOut, converter, stats)
            print("done")

        pathOutConverter = os.path.join(args.pathDB, f"phonesMatches.json")
        with open(pathOutConverter, 'w') as file:
            json.dump(converter, file)

        pathOutStats = os.path.join(args.pathDB, f"phonesStats.json")
        with open(pathOutStats, 'w') as file:
            json.dump(stats, file)

    if args.command == 'remove_phone':

        pathList = extractSeqListFromTSV(
            os.path.join(args.pathDB, "train.tsv"))
        if args.min_occ > 0:
            pathOutStats = os.path.join(args.pathDB, f"phonesStats.json")
            with open(pathOutStats, 'rb') as file:
                statsData = json.load(file)

            tabPrint = [(k, val) for k, val in statsData.items()]
            tabPrint.sort(key=lambda x: x[1])
            print(tabPrint)

            forbidden = [x for x in statsData if statsData[x] < args.min_occ]
            if len(forbidden) > 0:
                if args.invalidPhones is None:
                    args.invalidPhones = forbidden
                else:
                    args.invalidPhones = args.invalidPhones + forbidden

        if args.invalidPhones is not None:
            print(f"Removing {args.invalidPhones}")
            with open(os.path.join(args.pathDB, f"phonesMatches.json"), 'rb') as file:
                converter = json.load(file)
            newPhoneList = {v for k, v in converter.items()
                            if k not in args.invalidPhones}
            newPhoneConverter, backwardConverter = getNewPhoneConverter(
                newPhoneList, converter)
            #newPhoneDict = applyBackwardConverter(newPhoneDict, backwardConverter)

            # print(newPhoneConverter)

            with open(os.path.join(args.pathDB, f"phonesMatches_reduced.json"), 'w') as file:
                json.dump(newPhoneConverter, file)

            targets = ["train", "dev", "test", "validated"]
            fullDict = {}
            for name in targets:
                locPath = os.path.join(args.pathDB,
                                       f"{name}_phones.txt")
                locDict = loadPhoneDict(locPath)
                locDict = removeSeqsNotWithTranscription(locDict,
                                                         backwardConverter)
                locDict = applyBackwardConverter(locDict,
                                                 backwardConverter)
                savePhoneDict(locDict,
                              os.path.join(args.pathDB,
                                           f"{name}_phones_reduced.txt"))
                for k in locDict:
                    fullDict[k] = locDict[k]
        else:
            targets = ["train", "dev", "test"]
            for name in targets:
                locPath = os.path.join(args.pathDB,
                                       f"{name}_phones.txt")
                locDict = loadPhoneDict(locPath)
                for k in locDict:
                    fullDict[k] = locDict[k]
                savePhoneDict(locDict,
                              os.path.join(args.pathDB,
                                           f"{name}_phones_reduced.txt"))

    if args.command == 'make':

        filter_validated = extractSeqListFromTSV(
            str(Path(args.pathDB) / "validated.tsv"))
        pathOut16k = str(Path(args.pathDB) / "clips_16k")

        if args.to_16k:
            print("Resamplimg to 16kHz...")
            if Path(pathOut16k).is_dir():
                shutil.rmtree(pathOut16k)

            if not os.path.isdir(pathOut16k):
                os.mkdir(pathOut16k)

            path_in_clips = str(Path(args.pathDB) / "clips")
            all_seqs = [x for x in os.listdir(
                path_in_clips) if os.path.splitext(x)[1] == '.mp3']
            locList = get_common_seqs(all_seqs, filter_validated)
            adjustSampleRate(path_in_clips,
                             locList, pathOut16k)
            print("done")

        all_seqs = [x[1] for x in findAllSeqs(pathOut16k,
                                              extension='.mp3')[0]]
        pathList = get_common_seqs(all_seqs, filter_validated)

        if args.gain != 1.:
            print(f"Applying a gain of {args.gain}")
            pathOut16k = os.path.join(args.pathDB, "clips_16k")
            applyGain(pathOut16k, pathOut16k, pathList, args.gain)

        if args.spot is not None:
            pathPhoneMatch = os.path.join(args.pathDB,
                                          f"phonesMatches_reduced.json")
            with open(pathPhoneMatch, 'rb') as file:
                phoneMatch = json.load(file)

            phoneIndex = phoneMatch[args.spot]

            pathPhoneDict = os.path.join(args.pathDB,
                                         f"validated_phones_reduced.txt")
            phoneDict = loadPhoneDict(pathPhoneDict)
            samples = spotKSampleWithPhonem(phoneDict, phoneIndex, k=3)

        if args.to_cpc:
            speaker_data = getSpeakerDataFromTSV(
                Path(args.pathDB) / "validated.tsv")
            to_cpc_org(speaker_data, pathOut16k, '.mp3')

    if args.command == 'get_filter':
        filter_validated = loadPhoneDict(
            str(Path(args.pathDB) / "validated_phones_reduced.txt"))
        path_out = Path(args.pathDB) / "all_seqs.txt"
        saveSeqList(path_out, list(filter_validated.keys()))

    if args.command == 'lookup':
        lookForPhonem(args.phone_transcript, args.index)

    if args.command == 'show_phones':
        path_phones = os.path.join(args.pathDB, 'phonesMatches.json')
        path_phones_stats = os.path.join(args.pathDB, 'phonesStats.json')

        if args.reduced:
            path_phones = os.path.join(
                args.pathDB, 'phonesMatches_reduced.json')

        with open(path_phones_stats, 'rb') as file:
            phones_stats = json.load(file)

        with open(path_phones, 'rb') as file:
            phone_list = json.load(file).keys()

        stats_couples = list(phones_stats.items())
        stats_couples.sort(key=lambda x: x[1], reverse=True)

        print("Valid phones")
        for phone, stat in stats_couples:
            if phone in phone_list:
                print(f"{phone} {stat}")

        print("Removed phones")
        for phone, stat in stats_couples:
            if phone not in phone_list:
                print(f"{phone} {stat}")
