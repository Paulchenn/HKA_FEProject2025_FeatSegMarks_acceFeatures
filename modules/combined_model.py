# import torch.nn as nn
# import torch.nn.functional as F
# from modules.iccv_Generator import ICCVGenerator

# class CombinedModel(nn.Module):
#     def __init__(self, weights_path, target_size=(480, 480)):
#         super().__init__()
#         self.gen = ICCVGenerator()      # Generator initialisieren
#         self.gen.load_weights(f"{weights_path}/tuned_G_119.pth")        # Vorgefertigte Gewichte in den Generator laden
#         self.target_size = target_size

#     def forward(self, z_input, label_img, context_img):
#         x_gen = self.gen(z_input, label_img, context_img)       # Generator aufrufen mit Noise, Label und Kontextbild
#         x_up = F.interpolate(
#             x_gen, size=self.target_size, mode='bilinear', align_corners=False
#         )
#         return x_up  # Nur das generierte Bild zurückgeben

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from modules.iccv_Generator import ICCVGenerator
from modules.xfeat import XFeat  # oder relativer Import, falls nötig

class CombinedModel(nn.Module):
    def __init__(self, weights_path, target_size=(480, 480), top_k=3000):
        super().__init__()
        self.gen = ICCVGenerator()
        self.gen.load_weights(f"{weights_path}/tuned_G_119.pth")
        self.target_size = target_size

        # XFeat-Modell laden (ggf. mit pretrained=False wenn du trainieren willst)
        self.xfeat = XFeat(top_k=top_k)

    def forward(self, z_input, label_img, context_img):
        # 1. Generator erzeugt deformiertes Bild
        x_gen = self.gen(z_input, label_img, context_img)

        # 2. Upsampling auf gewünschte Auflösung
        x_up = F.interpolate(x_gen, size=self.target_size, mode='bilinear', align_corners=False)

        # 3. XFeat auf das generierte Bild anwenden (batchweise!)
        features = self.xfeat.detectAndCompute(x_up)[0]

        return features, x_up
    
    def preprocess_for_generator(self, img):
        """
        Skaliert und normalisiert ein RGB-Bild für den Generator (auf 32×32, [-1,1]-Skalierung).
        Erwartet: NumPy (H,W,3) oder torch.Tensor (1,3,H,W)
        Gibt zurück: torch.Tensor (1,3,32,32)
        """
        if isinstance(img, np.ndarray):
            img = torch.tensor(img).permute(2, 0, 1).float() / 255.0  # (3,H,W)

        if img.ndim == 3:
            img = img.unsqueeze(0)  # (1,3,H,W)

        img = F.interpolate(img, size=(32, 32), mode='bilinear', align_corners=False)

        # Normalisierung auf [-1, 1]
        mean = torch.tensor([0.5, 0.5, 0.5], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=img.device).view(1, 3, 1, 1)
        img = (img - mean) / std

        return img

    def match_generated(self, img_real):
        """
        Führt Matching zwischen realem Bild und Generator-Ausgabe durch:
        - Verwendet Generator + XFeat intern (gekoppelt)
        - Gibt gematchte Keypoints (mkpts0, mkpts1) und das Generatorbild zurück
        """
        device = self.xfeat.dev  # Stelle sicher, dass alles auf demselben Device ist

        # 1. Eingabe vorbereiten
        z = torch.randn(1, 100, 1, 1).to(device)
        label_img = self.preprocess_for_generator(img_real).to(device)
        context_img = self.preprocess_for_generator(img_real).to(device)

        # 2. Generator erzeugt deformiertes Bild + XFeat extrahiert Features
        features_gen, img_gen = self.forward(z, label_img, context_img)

        # 3. Auch auf dem Originalbild Features extrahieren
        real_input = self.xfeat.parse_input(img_real)
        features_real = self.xfeat.detectAndCompute(real_input)[0]

        # 4. Match Deskriptoren
        idx0, idx1 = self.xfeat.match(features_real['descriptors'],
                                      features_gen['descriptors'])

        mkpts0 = features_real['keypoints'][idx0].cpu().numpy()
        mkpts1 = features_gen['keypoints'][idx1].cpu().numpy()

        return mkpts0, mkpts1, img_gen





# import torch.nn.functional as F
# import torch.nn as nn
# from modules.iccv_Generator import ICCVGenerator
# # from modules.iccv_descriptor import ICCVDescriptor
# # from modules.iccv_classificator import ICCVClassifier
# from modules.xfeat import XFeat

# class CombinedModel(nn.Module):
#     def __init__(self, weights_path, target_size=(480, 480)):
#         super().__init__()
#         #self.gen = ICCVGenerator()
#         #self.gen.load_weights(f"{weights_path}")

#         self.gen = ICCVGenerator()
#         #self.gen.load_weights("/app/code/tuned_G_119.pth") - unflexible Lösung
#         self.gen.load_weights(f"{weights_path}/tuned_G_119.pth")


#         self.desc = ICCVDescriptor()
#         self.desc.load_weights(f"{weights_path}/tuned_D_119.pth")

#         self.clf = ICCVClassifier()
#         self.clf.load_weights(f"{weights_path}/best_cls.pth")

#         self.xfeat = XFeat()

#         self.target_size = target_size  # z. B. (480, 480)

#     def forward(self, x):
#         # Originalgröße merken, falls du sie brauchst
#         orig_size = x.shape[-2:]

#         # Generator
#         x_gen = self.gen(x)  # typischerweise [B, 3, 32, 32]

#         # Upscaling auf gewünschte Größe
#         x_up = F.interpolate(
#             x_gen,
#             size=self.target_size,
#             mode='bilinear',
#             align_corners=False
#         )  # Jetzt z. B. [B, 3, 480, 480]

#         # Deskriptor (auf Originalgröße oder x_gen)
#         features = self.desc(x_gen)

#         # Klassifikation
#         prediction = self.clf(features)

#         # xFeat läuft auf upgescaltem Bild
#         keypoints, descriptors = self.xfeat.extract(x_up)

#         return {
#             "prediction": prediction,
#             "keypoints": keypoints,
#             "descriptors": descriptors
#         }
