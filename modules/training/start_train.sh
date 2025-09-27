#!/usr/bin/env bash
set -Eeuo pipefail

MEGA=/home/docker/torch/data/xfeat_data/train_data/Megadepth
COCO=/home/docker/torch/data/xfeat_data/train_data/coco_20k
CKPT=/home/docker/torch/data/xfeat_test_paul
SDBOA=/home/docker/torch/code/SDbOA
WEIGHTS="$SDBOA/Result/abortedTraining02_03__MegaDepth__DS32__TSD0_035/MegaDepth/temp_netG/netG__stage2__Epoch_007.pth"
CONF="$SDBOA/Result/abortedTraining02_03__MegaDepth__DS32__TSD0_035/config_runtime.json"

args=(
  --training_type xfeat_default
  --megadepth_root_path "$MEGA"
  --synthetic_root_path "$COCO"
  --ckpt_save_path "$CKPT"
  --use_SDbOA
  --path_to_SDbOA_weights "$WEIGHTS"
  --path_to_SDbOA_config "$CONF"
)
python3 -m modules.training.train "${args[@]}"
