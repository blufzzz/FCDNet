#!/bin/bash
python train.py \
--logdir=./logs \
--config=./configs/v2v.yaml \
--experiment_comment='v2v_singlebox_bs5' # _featuresTSC
