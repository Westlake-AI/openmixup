#!/usr/bin/env bash
set -x

PYTHON=${PYTHON:-"python"}
CFG=$1
GPUS=$2
WORK_DIR=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PY_ARGS=${@:4} # "--checkpoint $CHECKPOINT --pretrained $PRETRAINED"
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

$PYTHON -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/extract.py $CFG --layer-ind "0,1,2,3,4" --work_dir $WORK_DIR \
    --launcher pytorch ${PY_ARGS}
