#!/bin/bash
python train_patchseg.py \
--logdir=./logs/logs_nG \
--config=./configs/patch_segmentation.yaml \
--experiment_comment='v2v128-IN_Balanced_segpatch_ps64_ppb500_nG_bs1_pbs6_pov0.8_SFL2-d0.999-g0-lr-1e-3-shuffle-AUG-XYZ' 

# --experiment_comment='v2v128-IN_Balanced_segpatch_ps64_ppb500_nG_bs1_pbs6_pov0.8_SFL-d0.9-g2-lr-1e-3-shuffle-AUG-XYZ-PREDSTACK' 

