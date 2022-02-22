#!/bin/bash
python train_tio_classification.py \
--logdir=./logs_clf \
--config=./configs/tio_patch_classification.yaml \
--experiment_comment='resnet3d_GRID_ps32_BRATS_bs1_pbs64_pov0.9_clf0.5_fcd0.5_lr-1e-3_AUG' 

# --experiment_comment='resnet3d_GRID_ps32_FCD_bs1_pbs64_pov0.9_clf0.5_fcd0.5_lr-1e-3_AUG'  


