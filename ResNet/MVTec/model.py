import torch
import torch.nn as nn
from torchvision import models

class MVTecResNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # ResNet50 fc input is 2048. We need 1 output for binary classification.
        self.backbone.fc = nn.Linear(2048, 1)

    def forward(self, x):
        return self.backbone(x)
