#!/usr/bin/env bash

set -x

CFG=$1
GPUS=$2
CHECKPOINT=$3
PY_ARGS=${@:4}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

WORK_DIR="$(dirname $CHECKPOINT)/"

# test
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/visualizations/vis_loss_landscape.py \
    $CFG \
    $CHECKPOINT \
    --work_dir $WORK_DIR --launcher="pytorch" ${PY_ARGS}
