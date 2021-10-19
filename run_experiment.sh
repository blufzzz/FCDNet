#!/bin/bash
python train_unet3d.py \
--logdir=./logs/v2v \
--config=./configs/v2v.yaml \
--experiment_comment='v2v' 