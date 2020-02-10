# CPC_audio

This code implements the Contrast Predictive Coding algorithm on audio data, as described in the paper [Unsupervised Pretraining Transfers well Across Languages](https://arxiv.org/abs/2002.02848). This is an unsupervised method to train audio features directly from the raw waveform.

Moreover, this code also implements all the evaluation metrics used in the paper:
- [ABX discriminability](https://zerospeech.com/2017/track_1.html)
- [Phone and speaker linear separability](https://arxiv.org/abs/1807.03748)
- Transfer learning on other languages, using the [common voices datasets](https://voice.mozilla.org/en/datasets)

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

## Custom architectures

The code allows you to train a wide range of architectures. For example, to train the CPC method as described in [Van Den Oord's paper](https://arxiv.org/abs/1807.03748) just run:

```bash
python cpc/train.py --pathDB $PATH_AUDIO_FILES --pathCheckpoint $PATH_CHECKPOINT_DIR --pathTrain $TRAINING_SET --pathVal $VAL_SET --file_extension $EXTENSION --normMode batchNorm --rnnMode linear
```

Or if you want to train a model with a FFD prediction network instead of a transformer:
```bash
python cpc/train.py --pathDB $PATH_AUDIO_FILES --pathCheckpoint $PATH_CHECKPOINT_DIR --pathTrain $TRAINING_SET --pathVal $VAL_SET --file_extension $EXTENSION --rnnMode ffd --schedulerRamp 10
```

The --schedulerRamp option add a learning rate ramp at the beginning of the training: it barely affects the performance of a model with a transformer predictor but is necessary with other models.

Launch cpc/train.py -h to see all the possible options.

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

### Cross lingual transfer

To begin download the common voices datasets [here](https://voice.mozilla.org/en/datasets), you will also need to download our phonem annotations and our train / val / test splits for each language [here](https://dl.fbaipublicfiles.com/cpc_audio/common_voices_splits.tar.gz). Then unzip your data at PATH_COMMON_VOICES.
Unfortunately, the audio files in common voices don't have the same sampling rate as in Librispeech. Thus you'll need to convert them into 16kH audio using the command:

```bash
DIR_CC=$PATH_COMMON_VOICES
for x in fr zh it ru nl sv es tr tt ky; do python cpc/eval/utils/adjust_sample_rate.py ${DIR_CC}/${x}/clips ${DIR_CC}/${x}/validated_phones_reduced.txt ${DIR_CC}/${x}/clips_16k; done
```

You can now run the experiments described in the paper. To begin, you must train the linear classifier. You will find below the instructions for the Spanish dataset: you can run the experiments on any other dataset in the same fashion.

#### Frozen features

To run the training on frozen features with the one hour dataset, just run:

```bash
python cpc/eval/common_voices_eval.py train $PATH_COMMON_VOICES/es/clips_16k $PATH_COMMON_VOICES/es/validated_phones_reduced.txt $CHECKPOINT_TO_TEST --pathTrain $PATH_COMMON_VOICES/es/trainSeqs_1.0_uniform_new_version.txt  --pathVal $PATH_COMMON_VOICES/es/trainSeqs_1.0_uniform_new_version.txt --freeze -o $OUTPUT_DIR
```

#### Fine tuning

The command is quite similar to run the fine-tuning experiments on the 5 hours dataset. For example in French you need to run:
```bash
python cpc/eval/common_voices_eval.py train $PATH_COMMON_VOICES/es/clips_16k $PATH_COMMON_VOICES/es/validated_phones_reduced.txt $CHECKPOINT_TO_TEST --pathTrain $PATH_COMMON_VOICES/es/trainSeqs_5.0_uniform_new_version.txt --pathVal $PATH_COMMON_VOICES/es/trainSeqs_5.0_uniform_new_version.txt --freeze -o $OUTPUT_DIR
```

#### PER

Once the training is done, you can compute the associated phone error rate (PER) on the test subset. To do so, just run:

```bash
python cpc/eval/common_voices_eval.py per $OUTPUT_DIR --pathVal $PATH_COMMON_VOICES/es/testSeqs_uniform_new_version.txt --pathPhone $PATH_COMMON_VOICES/es/validated_phones_reduced.txt
```

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

## Citations
Please consider citing this project in your publications if it helps your research.

```
@misc{rivire2020unsupervised,
    title={Unsupervised pretraining transfers well across languages},
    author={Morgane Rivière and Armand Joulin and Pierre-Emmanuel Mazaré and Emmanuel Dupoux},
    year={2020},
    eprint={2002.02848},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```

## License

CPC_audio is MIT licensed, as found in the LICENSE file.
