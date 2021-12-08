#!/bin/bash
python train.py \
--logdir=./logs \
--config=./configs/v2v.yaml \
--experiment_comment='v2v_whole_s128_bs1_GN_trim_DICE_TSC_AUG' # 
