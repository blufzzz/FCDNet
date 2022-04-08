#!/bin/bash
python train_seg.py \
--logdir=./logs_approach_comparison \
--config=./configs/segmentation.yaml \
--experiment_comment='v2v_whole_s128_bs1_GN_Dice-BCE250_lr1e-3_ALL-MNI152-AUG' 