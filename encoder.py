import torch
import torch.nn as nn
from torchvision import models


class VisionEncoder(nn.Module):
    def __init__(self, spatial=False):
        super().__init__()

        # Load ResNet-50 pretrained model on some 1.1 mil images holy damn
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        if spatial:
            # Drop avgpool + fc: keep the [B, 2048, 7, 7] spatial feature map,
            # used for the multi-token (49 visual tokens) ablation.
            self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        else:
            # Remove the final classification(we dont want classification) layer (fc outputs [1000])
            # Everything up to and including avgpool gives us [2048]
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        self.spatial = spatial

        # Freeze all weights — we never want gradients flowing into ResNet
        for param in self.encoder.parameters():
            param.requires_grad = False #freeze as resnet is already pretrained to have the best weights possible

    def forward(self, x):
        # x shape: [B, 3, 224, 224]
        features = self.encoder(x)

        if self.spatial:
            # [B, 2048, 7, 7] -> [B, 49, 2048]  (49 spatial visual tokens)
            b, c, h, w = features.shape
            return features.view(b, c, h * w).permute(0, 2, 1)

        features = features.squeeze(-1).squeeze(-1)  # [B, 2048]
        return features
