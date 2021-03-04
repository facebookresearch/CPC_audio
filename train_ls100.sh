#!/bin/bash

set -e
set -x

RVERB="-v --dry-run"
RVERB=""
CPC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAVE_DIR="$(
python - "$@" << END
if 1:
  import argparse
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--pathCheckpoint')
  args, _ = parser.parse_known_args()
  print(args.pathCheckpoint)
END
)"

mkdir -p ${SAVE_DIR}/code
rsync --exclude '.*' \
      --exclude data \
      --exclude pretrained_models \
      --exclude '__pycache__' \
      --exclude '*runs*' \
      --exclude '*.pyc' \
      --exclude '*.ipynb' \
      --filter=':- .gitignore' \
    $RVERB -lrpt $CPC_DIR/ ${SAVE_DIR}/code/

echo $0 "$@" >> ${SAVE_DIR}/out.txt
exec python -u cpc/train.py \
    --pathDB /pio/data/zerospeech2021/LibriSpeech-wav/train-clean-100 \
    --pathTrain /pio/scratch/2/jch/wav2vec/LibriSpeech100_labels_split/train_split.txt \
    --pathVal /pio/scratch/2/jch/wav2vec/LibriSpeech100_labels_split/test_split.txt \
    --file_extension .wav \
    "$@" 2>&1 | tee -ai ${SAVE_DIR}/out.txt
