"""ResNet-50 backbone for binary lesion classification."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet50


def _load_resnet50(pretrained: bool):
    try:
        from torchvision.models import ResNet50_Weights

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        return resnet50(weights=weights)
    except ImportError:
        return resnet50(pretrained=pretrained)


def build(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    model = _load_resnet50(pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_target_layer(model: nn.Module) -> nn.Module:
    return model.layer4
