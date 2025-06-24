# modules/iccv/classifier.py
import torch
import torch.nn as nn

class ICCVClassifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))

    def forward(self, x):
        return self.fc(x)