# modules/iccv/descriptor.py
import torch
import torch.nn as nn

class ICCVDescriptor(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, output_dim)
        )

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))


    def forward(self, x):
        return self.net(x)