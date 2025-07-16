# modules/iccv_Generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

# Generator-Modell für Bildgenerierung
class ICCVGenerator(nn.Module):
    def __init__(self, d=128):
        super(ICCVGenerator, self).__init__()

        # Rauschvektor
        self.deconv1_1 = nn.ConvTranspose2d(100, d*2, 8, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)

        # Labelbild wird auch leicht verarbeitet
        self.deconv1_2 = nn.ConvTranspose2d(10, d*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*2)

        # Kombi aus Noise, Label und Kontext
        self.deconv2 = nn.ConvTranspose2d(d*6, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)

        #Letzte Schicht: RGB-Ausgabe
        self.deconv4 = nn.ConvTranspose2d(d*2, 3, 4, 2, 1)

        # CNN für Labelbild
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3_1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)

        # CNN für Kontextbild
        self.conv1_1col = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_2col = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.maxpool1col = nn.MaxPool2d(kernel_size=2)
        self.conv2_1col = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2col = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.maxpool2col = nn.MaxPool2d(kernel_size=2)
        self.conv3_1col = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)

        self.self_att = MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)

    # Lädt gespeicherte Gewichte in das Modell
    def load_weights(self, path):
        #self.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Lade Gewichte von: {path}")
        state_dict = torch.load(path, map_location="cpu")
        print("Enthaltene Keys:", list(state_dict.keys())[:5])  # Vorschau
        self.load_state_dict(state_dict)

    def forward(self, input, label, bc):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))

        label = F.relu(self.conv1_1(label))
        label = F.relu(self.conv1_2(label))
        label = self.maxpool1(label)
        label = F.relu(self.conv2_1(label))
        label = F.relu(self.conv2_2(label))
        label = self.maxpool2(label)
        y = F.relu(self.conv3_1(label))

        bc = F.relu(self.conv1_1col(bc))
        bc = F.relu(self.conv1_2col(bc))
        bc = self.maxpool1col(bc)
        bc = F.relu(self.conv2_1col(bc))
        bc = F.relu(self.conv2_2col(bc))
        bc = self.maxpool2col(bc)
        yc = F.relu(self.conv3_1col(bc))

        x = torch.cat([x, y, yc], 1)

        x = F.relu(self.deconv2_bn(self.deconv2(x)))

        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        x_attended, _ = self.self_att(x_flat, x_flat, x_flat)
        x = x_attended.permute(0, 2, 1).view(B, C, H, W)

        # Finales Bild
        x = F.tanh(self.deconv4(x))
        return x


# modules/iccv/generator.py
# import torch
# import torch.nn as nn

# class ICCVGenerator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Beispielarchitektur – passe sie an dein Modell an
#         self.net = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 3, kernel_size=3, padding=1),
#             nn.Tanh()
#         )

#     def load_weights(self, path):
#         self.load_state_dict(torch.load(path, map_location="cpu"))

#     def forward(self, x):
#         return self.net(x)