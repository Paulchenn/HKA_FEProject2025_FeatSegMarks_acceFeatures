# modules/iccv/generator.py
import torch
import torch.nn as nn

class ICCVGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Beispielarchitektur – passe sie an dein Modell an
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))

    def forward(self, x):
        return self.net(x)