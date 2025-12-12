# model.py
import torch
import torch.nn as nn
from torchvision import models


class CXRModel(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = models.resnet50(weights="IMAGENET1K_V1")
        self.backbone.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        logits = self.backbone(x)
        anomaly_score = torch.sigmoid(logits).max(dim=1).values
        return logits, anomaly_score
