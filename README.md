# CPC_audio

## Setup instructions

The installation is a tiny bit involved due to the torch-audio dependency.

0/ Clone the repo:
`git clone git@github.com:facebookresearch/CPC_torch.git && cd CPC_torch`

1/ Install libraries which would be required for torch-audio https://github.com/pytorch/audio :
 * MacOS: `brew install sox`
 * Linux: `sudo apt-get install sox libsox-dev libsox-fmt-all`

2/ `conda env create -f environment.yml && conda activate cpc37`

3/ Run setup.py
`python setup.py develop`

You can test your installation with:
`nosetests -d`

### CUDA driver

This setup is given for CUDA 9.2 if you use a different version of CUDA then please change the version of cudatoolkit in environment.yml.
For more information on the cudatoolkit version to use, please check https://pytorch.org/

### Standard datasets

We suggest to train the model either on [Librispeech](http://www.openslr.org/12/) or [libri-light](https://github.com/facebookresearch/libri-light).


## How to run a session

To run a new training session, use:

```bash
python cpc/train.py --pathDB $PATH_AUDIO_FILES --pathCheckpoint $PATH_CHECKPOINT_DIR --pathTrain $TRAINING_SET --pathVal $VAL_SET --file_extension $EXTENSION
```

Where:
- $PATH_AUDIO_FILES is the directory containing the audio files. The files should be arranged as below:
```
PATH_AUDIO_FILES  
│
└───speaker1
│   └───...
│         │   seq_11.{$EXTENSION}
│         │   seq_12.{$EXTENSION}
│         │   ...
│   
└───speaker2
    └───...
          │   seq_21.{$EXTENSION}
          │   seq_22.{$EXTENSION}
```

Please note that each speaker directory can contain an arbitrary number of subdirectories: the speaker label will always be retrieved from the top one.

- $PATH_CHECKPOINT_DIR in the directory where the checkpoints will be saved
- $TRAINING_SET is a path to a .txt file containing the list of the training sequences (see [here](https://drive.google.com/drive/folders/1BhJ2umKH3whguxMwifaKtSra0TgAbtfb) for example)
- $VALIDATION_SET is a path to a .txt file containing the list of the validation sequences
- $EXTENSION is the extension of each audio file

## How to restart a session

To restart a session from the last saved checkpoint just run
```bash
python cpc/train.py --pathCheckpoint $PATH_CHECKPOINT_DIR
```
## How to run an evaluation session

All evaluation scripts can be found in cpc/eval/.

### Linear separability:

After training, the CPC model can output high level features for a variety of tasks. For an input audio file sampled at 16kHz, the provided baseline model will output 256 dimensional output features every 10ms. We provide two linear separability tests one for speaker, one for phonemes, in which a linear classifier is trained on top of the CPC features with aligned labels, and evaluated on a held-out test set.

Train / Val splits as well as phone alignments for librispeech-100h can be found [here](https://drive.google.com/drive/folders/1BhJ2umKH3whguxMwifaKtSra0TgAbtfb).


Speaker separability:

```bash
python cpc/eval/linear_separability.py $PATH_DB $TRAINING_SET $VAL_SET $CHECKPOINT_TO_LOAD --pathCheckpoint $PATH_CHECKPOINT
```

Phone separability:
```bash
python cpc/eval/linear_separability.py $PATH_DB $TRAINING_SET $VAL_SET $CHECKPOINT_TO_LOAD --pathCheckpoint $PATH_CHECKPOINT --pathPhone $PATH_TO_PHONE_LABELS
```

You can also concatenate the output features of several model by providing several checkpoint to the --load option. For example the following command line:

```bash
python cpc/eval/linear_separability.py -$PATH_DB $TRAINING_SET $VAL_SET model1.pt model2.pt --pathCheckpoint $PATH_CHECKPOINT
```

Will evaluate the speaker separability of the concatenation of the features from model1 and model2.


### ABX score:

You can run the ABX score on the [Zerospeech2017 dataset](https://zerospeech.com/2017/index.html). To begin, download the dataset [here](https://download.zerospeech.com/). Then run the ABX evaluation on a given checkpoint with:

```bash
python ABX.py from_checkpoint $PATH_CHECKPOINT $PATH_ITEM_FILE $DATASET_PATH --seq_norm --strict --file_extension .wav --out $PATH_OUT
```
Where:
- $PATH_CHECKPOINT is the path pointing to the checkpoint to evaluate
- $PATH_ITEM_FILE is the path to the .item file containing the triplet annotations
- $DATASET_PATH path to the directory containing the audio files
- $PATH_OUT path to the directory into which the results should be dumped
- --seq_norm normalize each batch of features across the time channel before computing ABX
- --strict forces each batch of features to contain exactly the same number of frames.

## torch hub

This model is also available via [torch.hub](https://pytorch.org/docs/stable/hub.html). For more details, have a look at hubconf.py.

## Running a grid-search over hyper-parameters (FAIR only)

This feature requires submitit, which can be installed by running:
`pip install git+ssh://git@github.com/fairinternal/submitit@master#egg=submitit`

Premption is not yet supported, hence it is advised to use either `dev` or `priority` partitions.

Running a grid-search is as simple as
```json
python grid_search.py --sweep=cpc/utils/small_grid.json --name=test --partition=dev
```
where `cpc/utils/small_grid.json` is a json file defining grid (as in [example](utils/small_grid.json)), `name` is the experiment name.
The resulting models and stdout/stderr streams of the runs would appear in `~/cpc/<name>/<data-time>/`.

You can use the `--dry_run` parameter which prevents the jobs from being actually launched.

## License

CPC_audio is MIT licensed, as found in the LICENSE file.
