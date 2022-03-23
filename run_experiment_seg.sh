#!/bin/bash
python train_segmentation.py \
--logdir=./logs_seg_features_comparison \
--config=./configs/segmentation.yaml \
--experiment_comment='v2v_whole_s128_bs1_GN_DICE_ALL_lr1e-3_trim-sulc' 