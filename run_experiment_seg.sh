#!/bin/bash
python train_seg.py \
--logdir=./logs/logs_nG \
--config=./configs/segmentation.yaml \
--experiment_comment='v2v128-IN_whole_s128_bs1_GN_Dice_lr1e-3_nG-AUG-autocast' 