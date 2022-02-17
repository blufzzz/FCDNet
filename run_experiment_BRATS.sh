#!/bin/bash
python train_tio_classification.py \
--logdir=./logs_BRATS_clf \
--config=./configs/tio_patch_classification.yaml \
--experiment_comment='resnet3d_ps32_BRATS_bs1_pbs16_clf0.5_fcd0.8_lr-1e-3_trim_GRID' 
