#!/bin/bash
python train_prec_classification.py \
--logdir=./logs_prec_clf \
--config=./configs/prec_patch_classification.yaml \
--experiment_comment='resnet3d_ps32_bs1_pbs52_clf0.5_fcd0.5_lr-5e-4_ALL_trim_pw1_BalRes_ep200-reshuffle' 

# python train.py \
# --logdir=./logs \
# --config=./configs/v2v.yaml \
# --experiment_comment='v2v128_whole_s128_bs1_GN_DICE_AUG_ALL_lr1e-3' 