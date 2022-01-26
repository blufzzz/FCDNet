#!/bin/bash
python train_prec_classification.py \
--logdir=./logs_prec_clf \
--config=./configs/prec_patch_classification.yaml \
--experiment_comment='resnet3d_ps52_bs1_pbs32_clf0.5_fcd0.5_ALL_trim_pw1_BalRes_ep200' 

# python train.py \
# --logdir=./logs \
# --config=./configs/v2v.yaml \
# --experiment_comment='v2v128_whole_s128_bs1_GN_DICE_AUG_ALL_lr1e-3' 