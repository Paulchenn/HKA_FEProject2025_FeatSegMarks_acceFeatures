import os
import pdb
import sys
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

# Project imports (reuse your training code)
from utils.modules import DS_EMSE, TSD
from utils.helpers import blur_image, blur_image_dynamic


class shapeDeformation:
    """
    Online texture/shape augmentation for training-time usage inside XFeat loop.

    Default behavior keeps **geometry** unchanged (Stage-1-like generation) so that
    your pre-computed correspondences remain **valid**. It repaints texture guided by edges
    and blends the generated image with the original. Optionally, you can enable true
    shape deformation (TPS) and point-warping later (TODO hooks provided).

    Call: y = shapeDeformation()(x)
      - x: Tensor [B,3,H,W], dtype float, range ~ [0,1] or [-1,1]
      - returns: Tensor [B,3,H,W] in the **same range** as input.

    Notes
    -----
    * If netG weights are not found, it falls back to identity (no-op) to avoid
      breaking your training.
    * To enable strong, geometry-changing TSD, set env SDbOA_USE_TSD=1 (but then
      you MUST also warp your correspondence points with the same grid; hooks are
      sketched but not wired by default here to keep Trainer changes zero).
    """

    def __init__(
            self,
            device,
            config,
            netG
    ):
        self.debug_prints = False

        # --- Device
        self.device = device

        # --- config
        self.config = config

        # --- Generator input resolution & noise dim (match SDbOA training)
        self.G_RES_HEIGHT = getattr(self.config, "image_height", 608)
        self.G_RES_WIDTH  = getattr(self.config, "image_width", 800)
        self.Z_DIM = getattr(self.config, "noise_size", 100)

        # --- Keep geometry by default (Stage-1-like). If you set this to 1 we'll apply TSD.
        self.tsd_lam = getattr(self.config, "tsd_lam", 0.035)

        # --- Ops
        self.ds_emse = DS_EMSE(self.config)
        self.tsd = TSD(self.config, device=device)

        # --- Build generator
        self.netG = netG
        self.netG.eval()

        # --- Preproc for generator: square center-crop -> resize -> normalize [-1,1]
        self._resize = transforms.Resize((self.G_RES_HEIGHT, self.G_RES_WIDTH))
        self._center_crop = None  # we handle square crop manually per batch

        self._warned_no_weights = False

    def _to_numpy_img(self, t: torch.Tensor):
        """
        Erwartet Tensor [B,C,H,W] oder [C,H,W] in [-1,1].
        Gibt float32-Numpy-Bild [H,W,3] in [0,1] zurück.
        """
        if t.dim() == 4:
            t = t[0]  # erstes Bild
        if t.size(0) == 1:
            t = t.repeat(3, 1, 1)  # Grau -> RGB

        # WICHTIG: float() zwingt fp32 (verhindert Matplotlib-Fehler)
        img = t.detach().float().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * 0.5 + 0.5, 0.0, 1.0)
        return img.astype(np.float32)

    # -----------------------
    # Public API (callable)
    # -----------------------
    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
        blend_alpha=1,  # 0..1; 0=no change, 1=full generated
        use_tsd=True,
        grid=None,
    ) -> torch.Tensor:
        """Return augmented tensor with same shape & range as input."""
        if not torch.is_tensor(x):
            raise TypeError('shapeDeformation expects a torch.Tensor [B,3,H,W].')
        assert x.dim() == 4 and x.shape[1] in (1, 3), 'Expected [B,C,H,W] with C in {1,3}.'

        B, C, H, W = x.shape
        x_dev = x.to(self.device)

        # Detect input range and create converters to preserve it
        x_min, x_max = float(x_dev.min().item()), float(x_dev.max().item())
        if x_min >= -1.05 and x_max <= 1.05:
            in_mode = 'minus1_1'  # [-1,1]
            x_n = x_dev.clone()
        else:
            in_mode = 'zero_1'    # assume [0,1] (or 0..255 already normalized upstream)
            x_n = (x_dev - 0.5) / 0.5

        # Ensure 3ch for the pipeline (generator expects RGB-like)
        if x_n.shape[1] == 1:
            x_rgb = x_n.repeat(1, 3, 1, 1)
        else:
            x_rgb = x_n

        # --- Prepare square input for netG to keep its receptive fields sane
        # Center-crop the shorter side to square and resize to G_RES
        crop = self._resize(x_rgb)
        x_dev = crop
        if self.debug_prints:
            print(f"x_dev shape: {x_dev.shape}")

        # --- Blur branch (I_txt)
        blur = blur_image(x_dev, downSize=getattr(self.config, "downSize2", 32))

        # --- Edge branch
        edge = self.ds_emse.diff_edge_map(x_dev)  # [B,3,R,R] in [0,1]
        
        # --- Optional shape deformation (disabled by default to keep labels valid)
        if use_tsd:
            edge_def, grid = self.tsd.doTSD(edge, grid=grid, return_grid=True, lam=self.tsd_lam)
        else:
            edge_def, grid = edge, None

        # --- Generate repaint using netG if weights available; else identity fallback
        if self.netG is not None:
            zdim = getattr(self.config, "noise_size", 100)
            z_ = Variable(torch.randn((B, zdim)).view(-1, 100, 1, 1).to(self.device))
            if self.debug_prints:
                print(f"[Deformer][call] shape z_: {z_.shape}")
                print(f"[Deformer][call] shape x_dev: {x_dev.shape}")
                print(f"[Deformer][call] shape blur: {blur.shape}")
                print(f"[Deformer][call] shape edge_def: {edge_def.shape}")

                fig = plt.figure(figsize=(10.5, 8))
                plt.subplot(1, 3, 1); plt.title('img'); plt.imshow(self._to_numpy_img(x_dev))
                plt.subplot(1, 3, 2); plt.title('blur img'); plt.imshow(self._to_numpy_img(blur))
                plt.subplot(1, 3, 3); plt.title('edge map'); plt.imshow(self._to_numpy_img(edge_def))
                plt.tight_layout()
                plt.show()       # show the window (non-blocking)
            gen = self.netG(z_, edge_def, blur)  # [-1,1]
        else:
            if not self._warned_no_weights:
                print('[SDbOA] No generator available. Returning input unchanged.')
                self._warned_no_weights = True
            gen = x_dev  # already in [-1,1]

        # # --- Resize back to original canvas size (no geometric warp by default)
        # gen_up = F.interpolate(gen, size=(side, side), mode='bilinear', align_corners=False)
        # # paste the center square back into full canvas (keep outer regions from original)
        # out_canvas = x_rgb.clone()
        # out_canvas[:, :, y0:y0 + side, x0:x0 + side] = (
        #     blend_alpha * gen_up + (1.0 - blend_alpha) * x_rgb[:, :, y0:y0 + side, x0:x0 + side]
        # )

        # # --- Convert back to original channel count and range
        # y_rgb = gen
        # if C == 1:
        #     # return a 3-channel image to keep the Trainer's subsequent mean(1,keepdim=True) valid
        #     pass
        # # map range back if needed
        # if in_mode == 'zero_1':
        #     y_rgb = (y_rgb * 0.5) + 0.5
        # y = y_rgb.to(x.dtype).to(x.device)
        return gen, grid
