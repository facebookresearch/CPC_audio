#!/bin/bash

# Script for the Prometheus slurm cluster

set -x

RVERB=""  # =-v

REMOTE_USER=plgjch
REMOTE_HOST=pro.cyfronet.pl

# location of the main repository (contains data/)
CPC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REMOTE_CPC_DIR=/net/people/plgjch/scratch/CPC_audio
REMOTE_MINICONDA_DIR=/net/archive/groups/plggneurony/os/miniconda3
REMOTE_LIBRISPEECH_DIR=/net/people/plgjch/lscratch/plgjch/LibriSpeech-wav
REMOTE_LIBRISPEECH100_SPLITS=/net/archive/groups/plggneurony/data/librispeech/LibriSpeech100_labels_split

# top-level directory for experiments
REMOTE_EXPERIMENT_RUNDIR=/net/scratch/people/plgjch/cpc/

# adjust the main loop
# (it can go over .yaml files, over hyperparameters, etc.
for PARAMS in \
"--CPCCTCNumMatched 12 --nPredicts 8 --CPCCTCSkipEnd 0" \
"--CPCCTCNumMatched 12 --nPredicts 8 --CPCCTCSelfLoop --CPCCTCSkipBeg 1 --CPCCTCSkipEnd 0" \
"--CPCCTCNumMatched 12 --nPredicts 8 --CPCCTCSelfLoop --CPCCTCSkipBeg 1 --CPCCTCSkipEnd 3" \
; do

# low-level directory for experiments
EXP_TAG=remote_pro
PRINT_PARAMS = $(echo $PARAMS | tr -d ' ' | sed -e 's/-\+/_/g')
NAME=cpcctc${PRINT_PARAMS}
DIR=$EXP_TAG/$NAME
EXP_DIR=$REMOTE_EXPERIMENT_RUNDIR/$DIR

echo $EXP_DIR

continue

ssh -q $REMOTE_USER@$REMOTE_HOST mkdir -p $EXP_DIR

TMP_DIR=`mktemp -d`
mkdir $TMP_DIR/code
# symlink the data from the main dir

cat > $TMP_DIR/exp_train.sh <<EOF
#!/bin/bash -l
## Job name
#SBATCH -J ${EXP_TAG}_${NAME}
## Nodes
#SBATCH -N 1
## CPU per Node
#SBATCH -c 6
## GPU
#SBATCH --gres=gpu:2
##
#SBATCH --mem=40GB
##
#SBATCH --time=72:00:00
##
#SBATCH -A plgzerospeech2021gpu
##
#SBATCH -p plgrid-gpu
##
#SBATCH --output="$EXP_DIR/exp_%j.out"
##
#SBATCH --error="$EXP_DIR/exp_%j.out"

## go to the exp dir
cd "$EXP_DIR/code"

/bin/hostname

eval "\$($REMOTE_MINICONDA_DIR/bin/conda shell.bash hook)"
conda activate 202102-cpc

set -e
set -x

export PYTHONPATH=$EXP_DIR/code

python -u cpc/train.py \
    --pathCheckpoint $EXP_DIR \
    --pathDB ${REMOTE_LIBRISPEECH_DIR}/train-clean-100 --file_extension .flac \
    --pathTrain ${REMOTE_LIBRISPEECH100_SPLITS}/train_split.txt \
    --pathVal ${REMOTE_LIBRISPEECH100_SPLITS}/test_split.txt \
    --n_process_loader 1 --max_size_loaded 4000000000 --batchSizeGPU 32 \
    --normMode layerNorm --dropout --rnnMode transformer  --nLevelsGRU 2  \
    `#--schedulerRamp 10` --nEpoch 75 \
    --CPCCTC --limitNegsInBatch 8  --CPCCTCNumMatched 12  --nPredicts $NPREDS $SELFLOOP 

CP=$(ls $EXP_DIR/checkpoint*.pt | sed -e 's/.*_\([0-9]\+\).pt/\1/' | sort -n | tail -1)
mkdir -p $EXP_DIR/lineval_${CP}
python -u cpc/eval/linear_separability.py \
    ${REMOTE_LIBRISPEECH_DIR}/train-clean-100 \
    ${REMOTE_LIBRISPEECH100_SPLITS}/train_split.txt \
    ${REMOTE_LIBRISPEECH100_SPLITS}/test_split.txt \
    $EXP_DIR/checkpoint_${CP}.pt \
    --pathPhone ${REMOTE_LIBRISPEECH100_SPLITS}/converted_aligned_phones.txt \
    --file_extension .wav \
    --pathCheckpoint $EXP_DIR/lineval_${CP} \
    2>&1 | tee -ai $EXP_DIR/lineval_${CP}/out.txt
EOF

# Transmit the startup script
rsync $RVERB -lrpt -e "ssh -q" $TMP_DIR/ $REMOTE_USER@$REMOTE_HOST:$EXP_DIR/

# Transmit the rest
rsync --exclude '.*' \
      --exclude data \
      --exclude pretrained_models \
      --exclude '__pycache__' \
      --exclude '*runs*' \
      --exclude '*.pyc' \
      --exclude '*.ipynb' \
      --filter=':- .gitignore' \
    $RVERB -lrpt -e "ssh -q" $CPC_DIR/ $REMOTE_USER@$REMOTE_HOST:$EXP_DIR/code/

ssh -q $REMOTE_USER@$REMOTE_HOST sbatch \
    `#--gres="" --time=00:10:00 -p plgrid-testing` \
    $EXP_DIR/exp_train.sh

rm -Rf $TMP_DIR

done

echo "Queue status"
ssh -q $REMOTE_USER@$REMOTE_HOST squeue
