#!/usr/bin/env bash
set -Eeuo pipefail

MEGA1500=/home/docker/torch/data/xfeat_data/Mega1500
WEIGHTS_XFEAT=/home/docker/torch/data/xfeat_training_genMegaDepth_DS32_TSD0_035/xfeat_default_150000.pth
SDBOA=/home/docker/torch/code/SDbOA
WEIGHTS_SDBOA="$SDBOA/Result/completeTraining__megaDepth__DS32__TSD0_035/MegaDepth/best_netG/netG__stage2__Epoch_042.pth"
CONF="$SDBOA/Result/completeTraining__megaDepth__DS32__TSD0_035/config_runtime_3.json"

args=(
  --dataset-dir "$MEGA1500"
  --weights-path "$WEIGHTS_XFEAT"
  --use_SDbOA
  --path_to_SDbOA_weights "$WEIGHTS_SDBOA"
  --path_to_SDbOA_config "$CONF"
  --do_Deformation
)
python3 -m modules.eval.megadepth1500 "${args[@]}"
