"""MobileNetV3-Small backbone for binary lesion classification (CPU-friendly)."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def build(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    # MobileNetV3 classifier is Sequential:
    # Linear(lastconv -> last_channel), Hardswish, Dropout, Linear(last_channel -> classes)
    in_features = model.classifier[0].in_features
    last_channel = model.classifier[0].out_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, last_channel),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(last_channel, num_classes),
    )
    return model


def get_target_layer(model: nn.Module) -> nn.Module:
    return model.features[-1]
