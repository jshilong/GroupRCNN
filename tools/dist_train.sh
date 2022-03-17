#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
WORK_DIR=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py --seed 0 $CONFIG --work-dir=${WORK_DIR} --launcher pytorch ${@:3}
