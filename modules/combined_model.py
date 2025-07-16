import torch.nn as nn
import torch.nn.functional as F
from modules.iccv_Generator import ICCVGenerator

class CombinedModel(nn.Module):
    def __init__(self, weights_path, target_size=(480, 480)):
        super().__init__()
        self.gen = ICCVGenerator()      # Generator initialisieren
        self.gen.load_weights(f"{weights_path}/tuned_G_119.pth")        # Vorgefertigte Gewichte in den Generator laden
        self.target_size = target_size

    def forward(self, z_input, label_img, context_img):
        x_gen = self.gen(z_input, label_img, context_img)       # Generator aufrufen mit Noise, Label und Kontextbild
        x_up = F.interpolate(
            x_gen, size=self.target_size, mode='bilinear', align_corners=False
        )
        return x_up  # Nur das generierte Bild zurückgeben



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
