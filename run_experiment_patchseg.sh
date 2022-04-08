#!/bin/bash
python train_segpatch.py \
--logdir=./logs_approach_comparison \
--config=./configs/patch_segmentation.yaml \
--experiment_comment='v2v128_Balanced_segpatch_ps64_ppb500_FCD_bs1_pbs8_pov0.8_Dice-lr-1e-3-shuffle-ALL-batchnorm' 
