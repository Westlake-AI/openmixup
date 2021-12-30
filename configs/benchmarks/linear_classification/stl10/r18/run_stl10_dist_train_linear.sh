#!/usr/bin/env bash

# Usage:
#    bash configs/benchmarks/linear_classification/stl10/r18/run_stl10_dist_train_linear.sh $GPU_id $PORT $WEIGHT.pth

base_path="configs/benchmarks/linear_classification/stl10/r18/"
exp_cfg0=$base_path"r18_lr0_1_bs256_head1.py"
exp_cfg1=$base_path"r18_lr1_0_bs256_head1.py"
exp_cfg2=$base_path"r18_lr10_bs256_head1.py"

set -e
set -x

GPU_ID=$1
PORT_id=$2
WEIGHT=$3


if [ "$GPU_ID" == "" ] || [ "$PORT_id" == "" ]; then
    echo "ERROR: Missing arguments."
    exit
fi

if [ "$WEIGHT" == "" ]; then
    echo "train with random init ! ! !"
    # random init train
fi

if [ "$WEIGHT" != "" ]; then
    echo "normal linear-supervised training start..."
    # normal train with 3 random seeds {0,1,3}
    CUDA_VISIBLE_DEVICES=$GPU_ID PORT=$PORT_id bash benchmarks/dist_train_linear_1gpu_sd.sh $exp_cfg1 $WEIGHT 0
    CUDA_VISIBLE_DEVICES=$GPU_ID PORT=$PORT_id bash benchmarks/dist_train_linear_1gpu_sd.sh $exp_cfg1 $WEIGHT 1
    CUDA_VISIBLE_DEVICES=$GPU_ID PORT=$PORT_id bash benchmarks/dist_train_linear_1gpu_sd.sh $exp_cfg1 $WEIGHT 3
fi
