#!/bin/bash

# Script for the Prometheus slurm cluster

set -x

RVERB=""  # =-v

REMOTE_USER=plgjch
REMOTE_HOST=pro.cyfronet.pl

# location of the main repository (contains data/)
CPC_DIR=/pio/scratch/2/jch/wav2vec/CPC_audio #"$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REMOTE_CPC_DIR=/net/people/plgjch/scratch/CPC_audio
REMOTE_MINICONDA_DIR=/net/archive/groups/plggneurony/os/miniconda3

# top-level directory for experiments
REMOTE_EXPERIMENT_RUNDIR=/net/scratch/people/plgjch/cpc/

# adjust the main loop
# (it can go over .yaml files, over hyperparameters, etc.
for DUMMY in aa \
; do

# low-level directory for experiments
EXP_TAG=remote_pro
NAME=baseline_1gpu
DIR=$EXP_TAG/$NAME
EXP_DIR=$REMOTE_EXPERIMENT_RUNDIR/$DIR

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
#SBATCH -c 8
## GPU
#SBATCH --gres=gpu:1
##
#SBATCH --mem=64GB
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
set -e
set -x

cd "$EXP_DIR/code"

/bin/hostname

eval "\$($REMOTE_MINICONDA_DIR/bin/conda shell.bash hook)"
conda activate 202102-cpc
export PYTHONPATH=$EXP_DIR/code

python -u cpc/train.py \
    --pathCheckpoint $EXP_DIR \
    --pathDB /net/archive/groups/plggneurony/data/librispeech/LibriSpeech/train-clean-100 --file_extension .flac \
    --pathTrain /net/archive/groups/plggneurony/data/librispeech/LibriSpeech100_labels_split/train_split.txt \
    --pathVal /net/archive/groups/plggneurony/data/librispeech/LibriSpeech100_labels_split/test_split.txt \
    --n_process_loader 1 --max_size_loaded 4000000000 --batchSizeGPU 32 \
    --normMode layerNorm --dropout --rnnMode transformer  --nLevelsGRU 2  \
    --schedulerRamp 10 --nPredicts 12 \
    --CPCCTC --limitNegsInBatch 8  --CPCCTCNumMatched 12  --nPredicts 12

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
