"""Tests for multimodal dataset, model architectures, and explainability."""

from __future__ import annotations

import torch
import pytest

from ml.src.architectures.multimodal import build_clinical_model, build_multimodal_model
from ml.src.dataset import MultimodalLesionDataset
from ml.src.explainability import explain_clinical


def test_multimodal_dataset_sampling():
    ds = MultimodalLesionDataset(split="test")
    assert len(ds) > 0
    sample = ds[0]
    assert "image" in sample
    assert "clinical" in sample
    assert "label" in sample
    assert "group_id" in sample

    assert sample["image"].ndim == 3
    assert sample["image"].shape == (3, 224, 224)
    assert sample["clinical"].ndim == 1
    assert sample["label"] in (0, 1)


def test_multimodal_model_forward_and_gradients():
    model = build_multimodal_model(
        backbone_name="resnet50",
        clinical_dim=10,
        num_classes=2,
        pretrained=False,
    )
    img = torch.randn(2, 3, 224, 224)
    clin = torch.randn(2, 10)

    # Forward
    logits = model(img, clin)
    assert logits.shape == (2, 2)

    # Freeze image backbone (Stage 1)
    model.freeze_image_backbone()
    target_layer = model.get_gradcam_target_layer()
    assert not any(p.requires_grad for p in target_layer.parameters())
    assert any(p.requires_grad for p in model.clinical_encoder.parameters())
    assert any(p.requires_grad for p in model.fusion_head.parameters())

    # Unfreeze last blocks (Stage 2)
    model.unfreeze_last_blocks()
    assert any(p.requires_grad for p in target_layer.parameters())


def test_clinical_only_model():
    model = build_clinical_model(clinical_dim=12)
    clin = torch.randn(4, 12)
    logits = model(clin)
    assert logits.shape == (4, 2)


def test_explain_clinical():
    model = build_clinical_model(clinical_dim=4)
    model.eval()
    clin = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    feats = ["fever", "rash", "headache", "chills"]
    res = explain_clinical(model, clin, feats, target_class=1)
    assert "attributions" in res
    assert "top_supporting_symptoms" in res
    assert "top_opposing_symptoms" in res
    assert len(res["attributions"]) <= len(feats)
