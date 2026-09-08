"""Registry tests — build models with pretrained=False (no weight download)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from ml.src.models import (
    MODEL_NAMES,
    REGISTRY,
    build_model,
    count_parameters,
    freeze_backbone,
    get_gradcam_target_layer,
    unfreeze_last_blocks,
)


@pytest.mark.parametrize("name", list(MODEL_NAMES))
def test_build_model_pretrained_false(name):
    model = build_model(name, num_classes=2, pretrained=False, dropout=0.2)
    n_params = count_parameters(model)
    assert n_params > 0
    layer = get_gradcam_target_layer(model, name)
    assert isinstance(layer, torch.nn.Module)
    x = torch.randn(1, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 2)


def test_unknown_model_name_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("not_a_real_backbone", pretrained=False)


def test_registry_covers_contract_names():
    assert set(REGISTRY) == set(MODEL_NAMES)
    assert set(MODEL_NAMES) == {
        "efficientnet_b0",
        "densenet121",
        "resnet50",
        "mobilenet_v3_small",
    }


def test_freeze_backbone_and_unfreeze_last_blocks():
    name = "mobilenet_v3_small"
    model = build_model(name, pretrained=False, num_classes=2, dropout=0.1)
    freeze_backbone(model, name)
    head_trainable = any(p.requires_grad for p in model.classifier.parameters())
    backbone_frozen = not any(p.requires_grad for p in model.features.parameters())
    assert head_trainable
    assert backbone_frozen

    unfreeze_last_blocks(model, name)
    last = list(model.features.children())[-1]
    assert any(p.requires_grad for p in last.parameters())
    first = list(model.features.children())[0]
    assert not any(p.requires_grad for p in first.parameters())


def test_resnet_freeze_keeps_fc_trainable():
    model = build_model("resnet50", pretrained=False, num_classes=2, dropout=0.0)
    freeze_backbone(model, "resnet50")
    assert any(p.requires_grad for p in model.fc.parameters())
    assert not any(p.requires_grad for p in model.layer1.parameters())
    unfreeze_last_blocks(model, "resnet50")
    assert any(p.requires_grad for p in model.layer4.parameters())
    assert not any(p.requires_grad for p in model.layer1.parameters())
