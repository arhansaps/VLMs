import torch
import torch.nn as nn
from torchvision import models


class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Load ResNet-50 pretrained on ImageNet
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Remove the final classification layer (fc outputs [1000])
        # Everything up to and including avgpool gives us [2048]
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze all weights — we never want gradients flowing into ResNet
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x shape: [B, 3, 224, 224]
        features = self.encoder(x)       # [B, 2048, 1, 1]  (avgpool keeps spatial dims)
        features = features.squeeze(-1).squeeze(-1)  # [B, 2048]
        return features
