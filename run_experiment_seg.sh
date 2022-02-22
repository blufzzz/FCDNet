#!/bin/bash
python train_segmentation.py \
--logdir=./logs_seg \
--config=./configs/v2v.yaml \
--experiment_comment='v2v_whole_s128_bs1_GN_DICE_AUG_ALL_lr1e-3_trim' 