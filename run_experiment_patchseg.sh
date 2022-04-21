#!/bin/bash
python train_patchseg.py \
--logdir=./logs/logs_nG \
--config=./configs/patch_segmentation.yaml \
--experiment_comment='v2v128-IN_Balanced_segpatch_ps64_ppb500_nG_bs1_pbs8_pov0.8_SFL-d99-g2-lr-1e-3-shuffle-AUG-autocast-XYZ' 

# --experiment_comment='v2v128-IN_Balanced_segpatch_ps64_ppb500_nG_bs1_pbs8_pov0.8_UFd0.8g1.3l0.5-lr-1e-3-shuffle-AUG-autocast-XYZ' 
