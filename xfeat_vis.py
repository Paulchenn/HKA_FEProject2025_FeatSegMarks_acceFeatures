#!/usr/bin/env python3
"""
XFeat Visualizer (robust)
- Lädt zwei Bilder
- Initialisiert XFeat (optional mit custom weights)
- Rechnet sparse (xfeat) oder semi-dense (xfeat-star) Matches
- Visualisiert Keypoints & Matches als JPGs
- Fängt Zero-Feature-Fälle sauber ab

Beispiel:
python3 xfeat_vis.py --img1 a.jpg --img2 b.jpg --matcher xfeat --weights /path/to/weights.pth \
  --max-long-edge 1200 --top-k 4096 --min-cossim 0.2 --device cuda
"""

import argparse
import os
import cv2
import numpy as np
import torch
import json
import sys
from modules.xfeat import XFeat
from torchvision.transforms.functional import rotate

# ----------------------------- Utils -----------------------------

def imread_rgb(path, max_long_edge=None):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Kann Bild nicht lesen: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if max_long_edge is not None and max_long_edge > 0:
        h, w = img_rgb.shape[:2]
        L = max(h, w)
        if L > max_long_edge:
            s = max_long_edge / float(L)
            img_rgb = cv2.resize(img_rgb, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return img_rgb

def to_tensor(img_rgb):
    t = torch.from_numpy(img_rgb).float() / 255.0  # [H,W,3]
    t = t.permute(2, 0, 1).unsqueeze(0)            # [1,3,H,W]
    return t


def to_hwc_uint8(x):
    """Bringt x auf NumPy HWC uint8 (für Zeichnen/Visualisieren)."""
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        if t.ndim == 4:
            t = t[0]
        if t.ndim == 3 and t.shape[0] == 3:   # CHW -> HWC
            t = t.permute(1, 2, 0)
        a = t.numpy()
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 1)
            a = (a * 255.0).astype(np.uint8)
        return a
    # NumPy:
    a = np.asarray(x)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 1) if a.max() <= 1.0 else np.clip(a / 255.0, 0, 1)
        a = (a * 255.0).astype(np.uint8)
    return a

def ensure_numpy_xy(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Erwarte Nx2 Koordinaten, bekommen {arr.shape}")
    return arr.astype(np.float32)

def draw_keypoints(img_rgb, kpts, radius=2, thickness=1):
    vis = img_rgb.copy()
    for x, y in kpts:
        cv2.circle(vis, (int(round(x)), int(round(y))), radius, (0, 255, 0), thickness, lineType=cv2.LINE_AA)
    return vis

def hconcat_with_padding(img_left, img_right, pad_val=255):
    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]
    H = max(h1, h2)
    def pad(img, H):
        h, w = img.shape[:2]
        if h == H:
            return img
        pad = np.full((H - h, w, 3), pad_val, dtype=img.dtype)
        return np.vstack([img, pad])
    left = pad(img_left, H)
    right = pad(img_right, H)
    canvas = np.hstack([left, right])
    return canvas, w1, H

def draw_matches(img1_rgb, img2_rgb, kpts1, kpts2, max_draw=500, thickness=1):
    vis, w1, _ = hconcat_with_padding(img1_rgb, img2_rgb)
    n = min(len(kpts1), len(kpts2), max_draw)
    if n == 0:
        return vis
    vis = np.ascontiguousarray(vis, dtype=np.uint8)
    idx = np.random.choice(len(kpts1), size=n, replace=False)
    pts1 = kpts1[idx]
    pts2 = kpts2[idx].copy()
    pts2[:, 0] += w1
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        color = tuple(np.random.randint(0, 256, size=3).tolist())
        cv2.line(vis, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color, thickness, cv2.LINE_AA)
        cv2.circle(vis, (int(round(x1)), int(round(y1))), 2, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, (int(round(x2)), int(round(y2))), 2, color, -1, lineType=cv2.LINE_AA)
    return vis

def parse_args():
    ap = argparse.ArgumentParser(description="XFeat Visualizer (robust)")
    ap.add_argument("--img1", type=str, required=True, help="Pfad zu Bild 1")
    ap.add_argument("--img2", type=str, required=True, help="Pfad zu Bild 2")
    ap.add_argument("--weights", type=str, default=None, help="Pfad zu custom XFeat-Gewichten (.pth/.pt)")
    ap.add_argument("--matcher", type=str, choices=["xfeat", "xfeat-star"], default="xfeat",
                    help="Sparse (xfeat) oder semi-dense (xfeat-star)")
    ap.add_argument("--top-k", type=int, default=4096, help="Top-K Features für detectAndCompute")
    ap.add_argument("--min-cossim", type=float, default=0.0, help="min. Cosinus-Ähnlichkeit fürs Matching (z.B. 0.2)")
    ap.add_argument("--draw", type=int, default=500, help="Max. Anzahl gezeichneter Matches")
    ap.add_argument("--max-long-edge", type=int, default=1200, help="Resize lange Kante vor Inferenz (0 = kein Resize)")
    ap.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda", help="Rechen-Device")
    ap.add_argument("--outdir", type=str, default=".", help="Ausgabe-Ordner")
    ap.add_argument("--fallback-star", action="store_true",
                    help="Falls sparse keine Features liefert, automatisch xfeat-star versuchen")
    ap.add_argument('--use_SDbOA', action='store_true',
                        help='Usage of Prepipeline. If on "True" xFeat will train with pretrainied SDbOA.')
    ap.add_argument('--path_to_SDbOA', type=str, default='/home/docker/torch/code/SDbOA',
                        help='Path to RePo of SDbOA.')
    ap.add_argument('--path_to_SDbOA_weights', type=str, default='/home/docker/torch/code/SDbOA/Result',
                        help='Path to weights of SDbOA')
    ap.add_argument('--path_to_SDbOA_config', type=str, default='/home/docker/torch/code/SDbOA/Result',
                        help='Path to weights of SDbOA')
    return ap.parse_args()

def _rotate_to_landscape(x: torch.Tensor):
        """
        Wenn H>W -> rotiere 90° (CCW) in Landscape.
        Gibt (x_rot, was_rotated: bool) zurück.
        """
        B, C, H, W = x.shape
        is_rot = False
        if H > W:
            x = rotate(img=x, angle=90, expand=True)
            is_rot = True
        return x, is_rot

def _undo_rotate(x: torch.Tensor, was_rotated: bool):
    """
    Dreht ggf. 90° zurück (CW), falls vorher rotiert wurde.
    """
    if was_rotated:
        return rotate(img=x, angle=270, expand=True)
    return x

# ----------------------------- Main -----------------------------

def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    # Device
    use_cuda = (args.device == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.device == "cuda" and not use_cuda:
        print("[Hinweis] CUDA angefordert, aber nicht verfügbar – nutze CPU.")

    # Bilder laden
    img1_rgb = imread_rgb(args.img1, max_long_edge=(args.max_long_edge if args.max_long_edge > 0 else None))
    img2_rgb = imread_rgb(args.img2, max_long_edge=(args.max_long_edge if args.max_long_edge > 0 else None))

    # Bilder deformiern
    if args.use_SDbOA:
        with open(args.path_to_SDbOA_config, 'r') as f:
            SDbOA_config = json.load(f)

        # get image sizes
        img_h = int(getattr(SDbOA_config, "image_height", 608))
        img_w = int(getattr(SDbOA_config, "image_width", 800))

        # gen = SDbOA_model.generator(img_size=256, z_dim=100).to(device)
        gen    = SDbOA_model.generator(
            img_size=(img_h, img_w),
            z_dim=getattr(SDbOA_config, "noise_size", 100),
            decoder_relu=True
        ).to(device)
        try:
            checkpoint = torch.load(args.path_to_SDbOA_weights, map_location=device)
            gen.load_state_dict(checkpoint)
            print(f"Loaded SDbOA Generator-Checkpoints from {args.path_to_SDbOA_weights}.")
        except:
            print(f"Failed to load SDbOA Generator-Checkpoints from {args.path_to_SDbOA_weights}.")
        gen.eval()

        deformer = shapeDeformation(
            device = device,
            config = SDbOA_config,
            netG = gen
        )
        

    # XFeat initialisieren
    if args.matcher == "xfeat-star":
        xfeat = XFeat(weights=args.weights, top_k=max(args.top_k, 10000))
    else:
        if args.weights==None:
            xfeat = XFeat(weights=args.weights)
        else:
            xfeat = XFeat(weights=args.weights)

    # --- Features extrahieren (einmalig) ---
    t1 = to_tensor(img1_rgb).to(device)
    t2 = to_tensor(img2_rgb).to(device)

    # 1) if needed turn images to landscape
    t1_rot, t1_was_rot = _rotate_to_landscape(t1)
    t2_rot, t2_was_rot = _rotate_to_landscape(t2)

    # 2) do deformation
    t1_def, grid    = deformer(x=t1_rot, blend_alpha=1, use_tsd=False, grid=None)
    t2_def, _       = deformer(x=t2_rot, blend_alpha=1, use_tsd=False, grid=grid)

    # 3) turn back to original
    t1 = _undo_rotate(t1_def, t1_was_rot)
    t2 = _undo_rotate(t2_def, t2_was_rot)
    
    with torch.inference_mode():
        out1 = xfeat.detectAndCompute(t1, top_k=args.top_k)[0]
        out2 = xfeat.detectAndCompute(t2, top_k=args.top_k)[0]

    kpts1 = ensure_numpy_xy(out1["keypoints"])
    kpts2 = ensure_numpy_xy(out2["keypoints"])
    n1, n2 = len(kpts1), len(kpts2)
    print(f"[Info] Keypoints: img1={n1}, img2={n2}")

    # --- Keypoints visualisieren ---
    img1_vis = to_hwc_uint8(t1)
    img2_vis = to_hwc_uint8(t2)

    vis_kp1 = draw_keypoints(img1_vis, kpts1)
    vis_kp2 = draw_keypoints(img2_vis, kpts2)
    cv2.imwrite(os.path.join(args.outdir, "vis_keypoints_img1.jpg"), cv2.cvtColor(vis_kp1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.outdir, "vis_keypoints_img2.jpg"), cv2.cvtColor(vis_kp2, cv2.COLOR_RGB2BGR))

    # --- Edge-Case: keine Features ---
    if n1 == 0 or n2 == 0:
        print("[Warnung] In mindestens einem Bild wurden 0 Features gefunden – überspringe sparse Matching.")
        if args.fallback_star:
            print("[Hinweis] Fallback auf xfeat-star …")
            ret = xfeat.match_xfeat_star(t1, t2)
            # vereinheitlichen
            if isinstance(ret, (tuple, list)) and len(ret) == 2:
                mkpts0, mkpts1 = ret
                mkpts0 = ensure_numpy_xy(mkpts0)
                mkpts1 = ensure_numpy_xy(mkpts1)
            else:
                m = ret[0] if isinstance(ret, (tuple, list)) else ret
                if isinstance(m, torch.Tensor):
                    m = m.detach().cpu().numpy()
                m = np.asarray(m)
                if m.ndim != 2 or m.shape[1] != 4:
                    raise ValueError(f"Unerwartetes Format von match_xfeat_star: {type(ret)} / shape {m.shape}")
                mkpts0 = m[:, :2].astype(np.float32)
                mkpts1 = m[:, 2:].astype(np.float32)
        else:
            mkpts0 = np.zeros((0, 2), dtype=np.float32)
            mkpts1 = np.zeros((0, 2), dtype=np.float32)
    else:
        # --- Sparse Matching mit bereits extrahierten Deskriptoren ---
        desc1 = out1["descriptors"]
        desc2 = out2["descriptors"]
        if isinstance(desc1, np.ndarray):
            desc1 = torch.from_numpy(desc1)
        if isinstance(desc2, np.ndarray):
            desc2 = torch.from_numpy(desc2)
        # Auf dasselbe Device bringen wie XFeat intern nutzt
        desc1 = desc1.to(device)
        desc2 = desc2.to(device)

        # match(): Cosinus-Ähnlichkeit + Argmax mit optionaler min_cossim-Schranke
        with torch.inference_mode():
            idxs0, idxs1 = xfeat.match(desc1, desc2, min_cossim=args.min_cossim)

        # Indizes -> Koordinaten
        k1 = out1["keypoints"].detach().cpu().numpy().astype(np.float32)
        k2 = out2["keypoints"].detach().cpu().numpy().astype(np.float32)
        mkpts0 = k1[idxs0.detach().cpu().numpy()]
        mkpts1 = k2[idxs1.detach().cpu().numpy()]

    # --- Matches visualisieren ---
    vis_matches = draw_matches(img1_vis, img2_vis, mkpts0, mkpts1, max_draw=args.draw, thickness=1)
    cv2.imwrite(os.path.join(args.outdir, "vis_matches.jpg"), cv2.cvtColor(vis_matches, cv2.COLOR_RGB2BGR))

    print("Fertig!")
    print(f"- Keypoints Bild 1: {n1}  -> {os.path.join(args.outdir, 'vis_keypoints_img1.jpg')}")
    print(f"- Keypoints Bild 2: {n2}  -> {os.path.join(args.outdir, 'vis_keypoints_img2.jpg')}")
    print(f"- Gezeichnete Matches: min({len(mkpts0)}, {len(mkpts1)}, {args.draw}) -> {os.path.join(args.outdir, 'vis_matches.jpg')}")

if __name__ == "__main__":

    args = parse_args()
    print(args.path_to_SDbOA)

    if args.use_SDbOA:
        sys.path.insert(0, args.path_to_SDbOA)
        from models import generation_imageNet_V2_3 as SDbOA_model
        from modules.dataset.shapeDeformation import *

    main(args)
