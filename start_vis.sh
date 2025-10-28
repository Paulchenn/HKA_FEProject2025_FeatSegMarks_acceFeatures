#!/usr/bin/env bash
set -Eeuo pipefail

IMG1=~/torch/data/xfeat_data/test_paul/img_1.jpg
IMG2=~/torch/data/xfeat_data/test_paul/img_2.jpg
WEIGHTS_XFEAT=~/torch/data/xfeat_training_genMegaDepth_DS32_TSD0_035/xfeat_default_160000.pth
SDBOA=~/torch/code/SDbOA
WEIGHTS_SDbOA="$SDBOA/Result/completeTraining__megaDepth__DS32__TSD0_035/MegaDepth/best_netG/netG__stage2__Epoch_042.pth"
CONF="$SDBOA/Result/completeTraining__megaDepth__DS32__TSD0_035/config_runtime_3.json"

args=(
  --img1 "$IMG1"
  --img2 "$IMG2"
  --matcher xfeat
  --weights  "$WEIGHTS_XFEAT"
  --min-cossim 0.95
  --use_SDbOA
  --path_to_SDbOA "$SDBOA"
  --path_to_SDbOA_weights "$WEIGHTS_SDbOA"
  --path_to_SDbOA_config "$CONF"
)

python3 -m xfeat_vis "${args[@]}"



# --img1 code/accFeatures/dataset/Mega1500/megadepth_test_1500/Undistorted_SfM/0015/images/62688623_17b5de833a_o.jpg
# --img2 code/accFeatures/dataset/Mega1500/megadepth_test_1500/Undistorted_SfM/0015/images/62689091_76cdd0858b_o.jpg
# --matcher xfeat
# --weights code/accFeatures/weights/xfeat.pt
# --min-cossim 0.9
# --use_SDbOA
# --path_to_SDbOA code/SDbOA
# --path_to_SDbOA_weights code/SDbOA/Result/completeTraining__megaDepth__DS32__TSD0_035/MegaDepth/best_netG/netG__stage2__Epoch_042.pth
# --path_to_SDbOA_config code/SDbOA/Result/completeTraining__megaDepth__DS32__TSD0_035/config_runtime_3.json