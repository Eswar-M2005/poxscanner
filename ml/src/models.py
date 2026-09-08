"""Model registry — backbones live in ``ml.src.architectures``."""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

from ml.src.architectures import densenet, efficientnet, mobilenet, resnet

MODEL_NAMES = (
    "efficientnet_b0",
    "densenet121",
    "resnet50",
    "mobilenet_v3_small",
)

REGISTRY: dict[str, object] = {
    "efficientnet_b0": efficientnet,
    "densenet121": densenet,
    "resnet50": resnet,
    "mobilenet_v3_small": mobilenet,
}

_HEAD_ATTR = {
    "efficientnet_b0": "classifier",
    "densenet121": "classifier",
    "resnet50": "fc",
    "mobilenet_v3_small": "classifier",
}


def _require(name: str):
    key = name.strip().lower()
    module = REGISTRY.get(key)
    if module is None:
        known = ", ".join(MODEL_NAMES)
        raise ValueError(f"Unknown model {name!r}. Expected one of: {known}")
    return key, module


def build_model(
    name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Build a registered backbone with a 2-class (or ``num_classes``) head."""
    _key, module = _require(name)
    builder: Callable[..., nn.Module] = module.build  # type: ignore[attr-defined]
    return builder(num_classes=num_classes, pretrained=pretrained, dropout=dropout)


def get_gradcam_target_layer(model: nn.Module, name: str) -> nn.Module:
    """Return the conv layer used for Grad-CAM for a registered architecture."""
    _key, module = _require(name)
    getter: Callable[[nn.Module], nn.Module] = module.get_target_layer  # type: ignore[attr-defined]
    return getter(model)


def count_parameters(model: nn.Module) -> int:
    """Total parameter count (trainable and frozen)."""
    return int(sum(p.numel() for p in model.parameters()))


def _head_module(model: nn.Module, name: str) -> nn.Module:
    attr = _HEAD_ATTR.get(name)
    if attr is None or not hasattr(model, attr):
        raise ValueError(f"No classification head mapping for {name!r}")
    return getattr(model, attr)


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad = enabled


def freeze_backbone(model: nn.Module, name: str) -> None:
    """Freeze all backbone weights; keep the classification head trainable."""
    key, _ = _require(name)
    _set_requires_grad(model, False)
    _set_requires_grad(_head_module(model, key), True)


def _last_backbone_modules(model: nn.Module, name: str) -> list[nn.Module]:
    if name == "resnet50":
        return [model.layer4]
    if name == "densenet121":
        modules = [model.features.denseblock4]
        if hasattr(model.features, "norm5"):
            modules.append(model.features.norm5)
        return modules
    if name in ("efficientnet_b0", "mobilenet_v3_small"):
        blocks = list(model.features.children())
        return blocks[-2:] if len(blocks) >= 2 else blocks
    return []


def unfreeze_last_blocks(model: nn.Module, name: str) -> None:
    """Unfreeze the last backbone stage(s) and the classification head."""
    key, _ = _require(name)
    for module in _last_backbone_modules(model, key):
        _set_requires_grad(module, True)
    _set_requires_grad(_head_module(model, key), True)


def head_parameters(model: nn.Module, name: str):
    """Parameters of the classification head."""
    key, _ = _require(name)
    return list(_head_module(model, key).parameters())


def trainable_backbone_parameters(model: nn.Module, name: str):
    """Trainable parameters that are not part of the classification head."""
    head_ids = {id(p) for p in head_parameters(model, name)}
    return [
        p
        for p in model.parameters()
        if p.requires_grad and id(p) not in head_ids
    ]
