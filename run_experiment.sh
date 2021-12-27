#!/bin/bash
python train_classification.py \
--logdir=./logs_clf \
--config=./configs/classification.yaml \
--experiment_comment='resnet1p2_ps32_bs1_TSG' 

# python train.py \
# --logdir=./logs \
# --config=./configs/v2v.yaml \
# --experiment_comment='v2v_whole_s128_bs1_GN_DICE_AUG_DUMMY_lr1e-3' 

