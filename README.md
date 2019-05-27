## Setup instructions

Check the setup [guide](setup/setup.MD).

## Repository architecture

train.py : main script

dataset.py : defintion of the Librispeech dataset

model.py : Basic encoders and AR models

criterion.py: definition of the training criterions. Three criterion are currently available: CPC (unsupervised), speaker classification and phone classification.

transformers.py: an implementation of transformers

unit_tests.py : unit tests

## How to run a session

```bash
python train.py --pathDB $PATH_TO_LIBRISPEECH_DB --pathCheckpoint $PATH_CHECKPOINT --pathTrain $TRAINING_SET --pathVal $VAL_SET
```


## How to run an evaluation session

Speaker separability:

```bash
python train.py --pathDB $PATH_TO_LIBRISPEECH_DB --pathCheckpoint $PATH_CHECKPOINT --supervised --eval --load $CHECKPOINT_TO_LOAD --pathTrain $TRAINING_SET --pathVal $VAL_SET
```

Phone separability:
```bash
python train.py --pathDB $PATH_TO_LIBRISPEECH_DB --pathCheckpoint $PATH_CHECKPOINT --supervised --eval --pathPhone $PATH_TO_PHONE_LABELS --load $CHECKPOINT_TO_LOAD --pathTrain $TRAINING_SET --pathVal $VAL_SET
```

You can also concatenate several model by providing several checkpoint to the --load option. For example the following command line:

```bash
python train.py --pathDB $PATH_TO_LIBRISPEECH_DB --supervised --eval --load model1.pt model2.pt
```

Will evaluate the speaker separability of the concatenation of model1 and model2.

## FAIR ONLY: reference arguments

Librispeech100 (clean):

--pathDB /datasets01/LibriSpeech/022219/train-clean-100/

--pathTrain /datasets01/LibriSpeech/022219/LibriSpeech100_labels_split/train_split.txt

--pathVal /datasets01/LibriSpeech/022219/LibriSpeech100_labels_split/test_split.txt

--pathPhone /private/home/mriviere/LibriSpeech/LibriSpeech100_labels_split/converted_aligned_phones.txt

Typical models:

/private/home/mriviere/FairInternal/CPC_torch/Librispeech100/uniform2 : uniform sampling

/private/home/mriviere/FairInternal/CPC_torch/Librispeech100/reverse : reverse model

/private/home/mriviere/FairInternal/CPC_torch/Librispeech100/2LevelsTrueLR : 2 levels GRU

/private/home/mriviere/FairInternal/CPC_torch/Librispeech100/full1284Speak_128Epoch: 4 speakers (sampling mode depreciated)

/private/home/mriviere/FairInternal/CPC_torch/Librispeech100/transformersCorrected : transformers AR

Librispeech360 (clean):

--pathDB /private/home/mriviere/libriSpeech360/LibriSpeech/train-clean-360

--pathTrain /private/home/mriviere/libriSpeech360/LibriSpeech/split_trainSeqs.txt

--pathVal /private/home/mriviere/libriSpeech360/LibriSpeech/split_valSeqs.txt

Librispeech500 (noisy):
--pathDB /private/home/mriviere/libriSpeech500/LibriSpeech/train-other-500

--pathTrain /private/home/mriviere/libriSpeech500/LibriSpeech/split_trainSeqs.txt

--pathVal /private/home/mriviere/libriSpeech500/LibriSpeech/split_valSeqs.txt
