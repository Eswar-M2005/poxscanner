"""DenseNet-121 backbone for binary lesion classification."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import densenet121


def _load_densenet121(pretrained: bool):
    try:
        from torchvision.models import DenseNet121_Weights

        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        return densenet121(weights=weights)
    except ImportError:
        return densenet121(pretrained=pretrained)


def build(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    model = _load_densenet121(pretrained)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_target_layer(model: nn.Module) -> nn.Module:
    return model.features.denseblock4
