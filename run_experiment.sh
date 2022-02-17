#!/bin/bash
python train_tio_classification.py \
--logdir=./logs_clf \
--config=./configs/tio_patch_classification.yaml \
--experiment_comment='resnet3d_ps32_bs1_pbs52_clf0.5_fcd0.5_lr-1e-3_ALL_trim' 