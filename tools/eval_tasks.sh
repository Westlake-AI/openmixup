#!/usr/bin/env bash


# Total Path -> work_dirs/classification/DATASET/METHOD/basic/BACKBONE/CKPT.pth
# Total Path -> configs/classification/DATASET/METHOD/basic/file.py
# 'work_dirs/classification/'
# PATH="work_dirs/classification"
# C_PATH="configs/classification"
# Checkpoint Path
CKPT=$1

# Config Path
CONFIG=$2

# 'work_dirs/total_eval_log'
WORK_DIR=$3

# Choose one of the heads
# [acc_mix_k, acc_one_k, acc_mix_q, acc_one_q] for automix, samix and adautomix, a4
# [head0] for general mixups
HEAD=$4

# Setting a max ratio for the occlusion experiments, please ensure the max ratio is between 0.0-1.0
CKPT_O=$5
CONFIG_O=$6
MAX_RATIO=$7

# This Config file for courrption evaluation
CONFIG_C=$8

# Task 1. FGSM compute adversarial robustness error
python tools/analysis_tools/calibration_fgsm.py \
        --checkpoint $CKPT \
        --config $CONFIG \
        --keys 'fgsm' \
        --head $HEAD \
        --dataset 'cifar' \
        --work_dir $WORK_DIR \

# Task 2. Calibration evaluation ECE
python tools/analysis_tools/calibration_fgsm.py \
        --checkpoint $CKPT \
        --config $CONFIG \
        --keys 'calibration' \
        --head $HEAD \
        --dataset 'cifar' \
        --work_dir $WORK_DIR \

# Task 3. Occlusion robustness experiments
python tools/analysis_tools/occlusion_robustness.py \
        --checkpoint $CKPT_O \
        --config $CONFIG_O \
        --max_ratio $MAX_RATIO \
        --work_dir $WORK_DIR \

# Task 4. CIFAR100-C test experiment  FIX: How to enforce the speed of load .npy files
python tools/analysis_tools/courrption.py \
        --checkpoint $CKPT \
        --config $CONFIG_C \
        --work_dir $WORK_DIR \

python tools/analysis_tools/merge_logs.py \
        --config $CONFIG \
        --work_dir $WORK_DIR \

