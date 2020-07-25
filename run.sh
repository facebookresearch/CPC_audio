#!/bin/bash
# CPC unsupervised extration of language features
source /home/zhangwq01/anaconda3/bin/activate torch_py37_yhb
echo $CONDA_DEFAULT_ENV

CUDA_VISIBLE_DEVICES=0

PATH_AUDIO_FILES='/home/zhangwq01/yuhaibin/data/olr18_train_cpc/data/cpc_train'
PATH_CHECKPOINT_DIR='/home/zhangwq01/yuhaibin/CPC_audio/LID_checkpoint/trial1'
TRAINING_SET='/home/zhangwq01/yuhaibin/data/olr18_train_cpc/data/train.txt'
VAL_SET='/home/zhangwq01/yuhaibin/data/olr18_train_cpc/data/val.txt'
EXTENSION='wav'
N_GPU=1
BATCHSIZE=48
N_EPOCH=100
STEP=5

python cpc/train.py --pathDB $PATH_AUDIO_FILES\
                    --pathCheckpoint $PATH_CHECKPOINT_DIR\
                    --pathTrain $TRAINING_SET\
                    --pathVal $VAL_SET\
                    --file_extension $EXTENSION\
                    --nGPU $N_GPU\
                    --batchSizeGPU $BATCHSIZE\
                    --nEpoch $N_EPOCH\
                    --save_step $STEP
echo 'done'
