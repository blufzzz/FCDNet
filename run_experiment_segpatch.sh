#!/bin/bash
python train_tio_segpatch.py \
--logdir=./logs_segpatch \
--config=./configs/tio_segpatch_classification.yaml \
--experiment_comment='v2v256_GRID_segpatch_ps64_FCD_bs1_pbs4_pov0.8_lr-1e-3_SINGLE-BRAIN' 

# --experiment_comment='resnet3d_GRID_segpatch_ps64_BRATS_bs1_pbs8_pov0.9_lr-1e-3' 