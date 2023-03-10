#!/bin/bash

T=`date +%m%d%H%M`
ROOT=../..
cfg=./resnet101_FWSNet_imagenet.yaml

export PYTHONPATH=$ROOT:$PYTHONPATH
            
python $ROOT/tools/trainval_dual.py \
--config $cfg \
--print_freq 100 \
--num_epoch 300 \
2>&1 | tee ./train.log
