#!/bin/bash
python train.py \
--logdir=./logs \
--config=./configs/v2v.yaml \
--experiment_comment='v2v128_whole_s128_bs1_GN_DICE_AUG_ALL_lr1e-3' 

# python train_classification.py \
# --logdir=./logs_clf \
# --config=./configs/classification.yaml \
# --experiment_comment='resnet1p2_ps64_bs1_pbs2_clf0.5_fcd0.8_ALL_AUG_trim_LS1-1' 

