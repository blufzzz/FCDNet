#!/bin/bash
python train_seg.py \
--logdir=./logs/logs_yarkin_data \
--config=./configs/segmentation.yaml \
--experiment_comment='v2v128red-IN_s128_bs1_SFL-g1-d0.9_lr1e-3_YARKIN-AUG-autocast' 

# python train_seg.py \
# --logdir=./logs/logs_yarkin_data \
# --config=./configs/segmentation.yaml \
# --experiment_comment='v2v128-IN_s128_bs1_Tversky-d0.9_lr1e-3_nG-AUG' 

# --experiment_comment='v2v128-IN_s128_bs1_Dice_lr1e-3_YARKIN-AUG-autocast' 

