## Repository architecture

train.py : main script

dataset.py : defintion of the Librispeech dataset

model.py : CPC model

criterion.py: definition of the training criterions. Two criterion are currently available: CPC (unsupervised) and speaker classification.

## How to run a session

```bash
python train.py --pathDB $PATH_TO_LIBRISPEECH_DB --pathCheckpoint $PATH_CHECKPOINT
```


## How to run an evaluation session

Speaker separability:

```bash
python train.py --pathDB $PATH_TO_LIBRISPEECH_DB --pathCheckpoint $PATH_CHECKPOINT --supervised --eval --load $CHECKPOINT_TO_LOAD
```

Phone separability:
```bash
python train.py --pathDB $PATH_TO_LIBRISPEECH_DB --pathCheckpoint $PATH_CHECKPOINT --supervised --eval --pathPhone $PATH_TO_PHONE_LABELS --load $CHECKPOINT_TO_LOAD
```
