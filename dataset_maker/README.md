# Dataset maker

## Preparing the common voices datasets

You will need to use the python script common_voice_db_maker.py

### Step one : download the dataset

Build a directory LANG_DIR where you will download and decompress the dataset.
```
mkdir $LANG_DIR; cd $LANG_DIR
wget $DOWNLOAD_URL
tar xvzf LANG_CODE.tar.gz
```

### Step two : launch the phone transcription

Get the phone transcription using phonemizer.
```
python common_voice_db_maker.py prepare_phone $LANG_DIR $LANG_CODE
```

You can see the statistics on the detected phones using the following command:
```
python common_voice_db_maker.py show_phones $LANG_DIR
```

To remove sequences with underrepresented phones run:
```
python common_voice_db_maker.py remove_phone $LANG_DIR -m $MIN_OCC_THRESHOLD
```

### Step three : resample the sequences to 16kHz

The sequences of the Common Voices dataset all have a 48kHz sample rate.
This is far too high for CPC which requires 16kHz inputs.
To resample the sequences to 16kHz just launch:
```
python common_voice_db_maker.py make --to_16k --to_cpc $LANG_DIR
```

### Step four : list all valid sequences

Several sequences have been discarded in step two.
To build a list of the valid sequences usable by other scripts in this codebase please launch:
```
python common_voice_db_maker.py get_filter $LANG_DIR
```

## Building train / val / dev / test split on any dataset

To begin, start by building the statistics on the given dataset
```
python get_all_stats.py $DATASET_DIR \
--file_extension $AUDIO_EXTENSION \
-o $OUTPUT_DIR \
--path_filter $FILES_WITH_THE_SEQUENCES_TO_SELECT
```

For a Common Voices dataset the command will be:
```
python get_all_stats.py $LANG_DIR/clips_16k \
--file_extension .mp3 \
-o $LANG_DIR/stats \
--path_filter $LANG_DIR/all_seqs.txt
```

Then build the train / val / dev / test
```
python make_db_split.py $LANG_DIR/stats \
--file_extension .mp3 \
-o $LANG_DIR/split \
--target_time $HOURS_TRAINING_DATASET \
--target_time_val $HOURS_VALIDATION_DATASET \
--target_time_dev 1 \
--target_time_test 1
```

You can use the parameter   --ignore [IGNORE [IGNORE ...]] to exclude a given list of sequences.
For example if already have a dev and test subsets you can exclude them with:

```
python make_db_split.py $LANG_DIR/stats \
--file_extension .mp3 \
-o $LANG_DIR/split \
--target_time $HOURS_TRAINING_DATASET \
--target_time_val $HOURS_VALIDATION_DATASET \
--ignore $PATH_DEV $PATH_TEST
```
