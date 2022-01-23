#!/bin/bash
python train_classification.py \
--logdir=./logs_clf \
--config=./configs/classification.yaml \
--experiment_comment='resnet3d_ps32_bs1_pbs10_clf0.5_fcd0.8_ALL_AUG_trim_LS1-ratio' 

# python train.py \
# --logdir=./logs \
# --config=./configs/v2v.yaml \
# --experiment_comment='v2v128_whole_s128_bs1_GN_DICE_AUG_ALL_lr1e-3' 