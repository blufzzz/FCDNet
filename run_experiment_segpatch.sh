#!/bin/bash
python train_tio_segpatch.py \
--logdir=./logs_approach_comparison \
--config=./configs/tio_patch_segmentation.yaml \
--experiment_comment='v2v256_GRID_segpatch_ps64_FCD_bs1_pbs4_pov0.8_DICE-BCE250-lr-1e-3-shuffle-ALL-AUG' 

# --experiment_comment='resnet3d_GRID_segpatch_ps64_BRATS_bs1_pbs8_pov0.9_lr-1e-3' 