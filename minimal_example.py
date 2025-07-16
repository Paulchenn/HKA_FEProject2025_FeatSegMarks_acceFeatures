"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Minimal example of how to use XFeat.
"""

import numpy as np
import os
import torch
import tqdm
import cv2

from modules.combined_model import CombinedModel
from modules.xfeat import XFeat

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU, comment for GPU

# Initialisiere XFeat und Generator
xfeat = XFeat()
weights_path = "/app/code"  # Passe den Pfad ggf. an
generator = CombinedModel(weights_path)

# ------------------------------
# Abschnitt 1: Generator-Ausgabe + XFeat
# ------------------------------

# Dummy-Eingaben für den Generator
z_input = torch.randn(1, 100, 1, 1)
label_img = torch.randn(1, 3, 32, 32)
context_img = torch.randn(1, 3, 32, 32)

with torch.no_grad():
    gen_img = generator(z_input, label_img, context_img)  # Ausgabe: [1,3,32,32]

# Generator-Ausgabe auf VGA skalieren
gen_img_resized = torch.nn.functional.interpolate(gen_img, size=(480, 640), mode='bilinear', align_corners=False)

# XFeat auf generiertem Bild anwenden
gen_output = xfeat.detectAndCompute(gen_img_resized, top_k=1024)[0]
print("===> XFeat on Generator Output")
print("keypoints:   ", gen_output['keypoints'].shape)
print("descriptors: ", gen_output['descriptors'].shape)
print("scores:      ", gen_output['scores'].shape)
print("----------------\n")

# ------------------------------
# VISUALISIERUNG: Keypoints auf dem Generatorbild
# ------------------------------

# Generatorbild in NumPy konvertieren
img = gen_img_resized.squeeze(0).permute(1, 2, 0).cpu().numpy()
img = (img + 1.0) / 2.0  # [-1,1] → [0,1]
img = (img * 255).astype(np.uint8)

# XFeat Keypoints in OpenCV-Format
kpts = gen_output['keypoints'].cpu().numpy()
#cv_kpts = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), _size=1) for p in kpts]
cv_kpts = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in kpts]


# Bild anzeigen
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
out_img = cv2.drawKeypoints(img_bgr, cv_kpts, None, color=(0, 255, 0))

cv2.imshow("Generator Output + XFeat Keypoints", out_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------------
# Abschnitt 2: Originale XFeat Tests (wie gehabt)
# ------------------------------

# Random input
x = torch.randn(1, 3, 480, 640)
output = xfeat.detectAndCompute(x, top_k=4096)[0]
print("===> XFeat on Random Image")
print("keypoints:   ", output['keypoints'].shape)
print("descriptors: ", output['descriptors'].shape)
print("scores:      ", output['scores'].shape)
print("----------------\n")

# Stress test
x = torch.randn(1, 3, 480, 640)
for i in tqdm.tqdm(range(100), desc="Stress test on VGA resolution"):
    output = xfeat.detectAndCompute(x, top_k=4096)

# Batched mode
x = torch.randn(4, 3, 480, 640)
outputs = xfeat.detectAndCompute(x, top_k=4096)
print("# detected features on each batch item:", [len(o['keypoints']) for o in outputs])

# Match two images with sparse features
x1 = torch.randn(1, 3, 480, 640)
x2 = torch.randn(1, 3, 480, 640)
mkpts_0, mkpts_1 = xfeat.match_xfeat(x1, x2)

# Match two images with semi-dense approach -- batched mode with batch size 4
x1 = torch.randn(4, 3, 480, 640)
x2 = torch.randn(4, 3, 480, 640)
matches_list = xfeat.match_xfeat_star(x1, x2)
print("Match shape (first pair):", matches_list[0].shape)


# import numpy as np
# import os
# import torch
# import tqdm

# from modules.xfeat import XFeat

# os.environ['CUDA_VISIBLE_DEVICES'] = '' #Force CPU, comment for GPU

# xfeat = XFeat()

# #Random input
# x = torch.randn(1,3,480,640)

# #Simple inference with batch = 1
# output = xfeat.detectAndCompute(x, top_k = 4096)[0]
# print("----------------")
# print("keypoints: ", output['keypoints'].shape)
# print("descriptors: ", output['descriptors'].shape)
# print("scores: ", output['scores'].shape)
# print("----------------\n")

# x = torch.randn(1,3,480,640)
# # Stress test
# for i in tqdm.tqdm(range(100), desc="Stress test on VGA resolution"):
# 	output = xfeat.detectAndCompute(x, top_k = 4096)

# # Batched mode
# x = torch.randn(4,3,480,640)
# outputs = xfeat.detectAndCompute(x, top_k = 4096)
# print("# detected features on each batch item:", [len(o['keypoints']) for o in outputs])

# # Match two images with sparse features
# x1 = torch.randn(1,3,480,640)
# x2 = torch.randn(1,3,480,640)
# mkpts_0, mkpts_1 = xfeat.match_xfeat(x1, x2)

# # Match two images with semi-dense approach -- batched mode with batch size 4
# x1 = torch.randn(4,3,480,640)
# x2 = torch.randn(4,3,480,640)
# matches_list = xfeat.match_xfeat_star(x1, x2)
# print(matches_list[0].shape)
