import torch.nn.functional as F

class CombinedModel(nn.Module):
    def __init__(self, weights_path, target_size=(480, 480)):
        super().__init__()
        self.gen = ICCVGenerator()
        self.gen.load_weights(f"{weights_path}/generator.pth")

        self.desc = ICCVDescriptor()
        self.desc.load_weights(f"{weights_path}/descriptor.pth")

        self.clf = ICCVClassifier()
        self.clf.load_weights(f"{weights_path}/classifier.pth")

        self.xfeat = XFeat()

        self.target_size = target_size  # z. B. (480, 480)

    def forward(self, x):
        # Originalgröße merken, falls du sie brauchst
        orig_size = x.shape[-2:]

        # Generator
        x_gen = self.gen(x)  # typischerweise [B, 3, 32, 32]

        # 📈 Upscaling auf gewünschte Größe
        x_up = F.interpolate(
            x_gen,
            size=self.target_size,
            mode='bilinear',
            align_corners=False
        )  # Jetzt z. B. [B, 3, 480, 480]

        # Deskriptor (auf Originalgröße oder x_gen)
        features = self.desc(x_gen)

        # Klassifikation
        prediction = self.clf(features)

        # xFeat läuft auf upgescaltem Bild
        keypoints, descriptors = self.xfeat.extract(x_up)

        return {
            "prediction": prediction,
            "keypoints": keypoints,
            "descriptors": descriptors
        }