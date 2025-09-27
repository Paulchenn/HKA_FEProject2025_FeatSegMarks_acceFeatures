#!/usr/bin/env bash
set -Eeuo pipefail

IMG1=/home/docker/torch/data/xfeat_data/test_paul/img_1.jpg
IMG2=/home/docker/torch/data/xfeat_data/test_paul/img_2.jpg
WEIGHTS_XFEAT=./weights/xfeat.pt
SDBOA=/home/docker/torch/code/SDbOA
WEIGHTS_SDbOA="$SDBOA/Result/20250911_172747/ImageNet256/best_netG/netG__stage2__Epoch_336.pth"
CONF="$SDBOA/Result/20250911_172747/config_runtime.json"

args=(
  --img1 "$IMG1"
  --img2 "$IMG2"
  --matcher xfeat
  --weights  "$WEIGHTS_XFEAT"
  --min-cossim 0.9
  --use_SDbOA
  --path_to_SDbOA "$SDBOA"
  --path_to_SDbOA_weights "$WEIGHTS_SDbOA"
  --path_to_SDbOA_config "$CONF"
)

python3 -m xfeat_vis "${args[@]}"