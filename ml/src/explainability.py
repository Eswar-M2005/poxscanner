"""Explainability module providing dual modality interpretability:

1. Visual localization via Grad-CAM for lesion images.
2. Clinical feature attributions via sensitivity/perturbation analysis for clinical symptoms.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ml.src.gradcam import GradCAM, cam_to_png_bytes, overlay_cam_on_image


def explain_image(
    model: nn.Module,
    image_tensor: torch.Tensor,
    pil_image: Image.Image,
    target_layer: nn.Module | None = None,
    target_class: int = 1,
) -> tuple[np.ndarray, bytes]:
    """Generate Grad-CAM heatmap array and overlay PNG bytes."""
    if target_layer is None:
        if hasattr(model, "get_gradcam_target_layer"):
            target_layer = model.get_gradcam_target_layer()
        else:
            raise ValueError("Target layer must be specified or available on model")

    # GradCAM expects single-image batch tensor with grad
    tensor = image_tensor.clone().detach()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor.requires_grad_(True)

    cam_engine = GradCAM(model, target_layer)
    # Forward hook
    heatmap = cam_engine.generate(tensor, target_class=target_class)

    # Blend on original image
    rgb_arr = np.array(pil_image.convert("RGB"))
    overlay_arr = overlay_cam_on_image(rgb_arr, heatmap, alpha=0.45)
    overlay_bytes = cam_to_png_bytes(overlay_arr)
    return heatmap, overlay_bytes


def explain_clinical(
    model: nn.Module,
    clinical_tensor: torch.Tensor,
    feature_names: Sequence[str],
    image_tensor: torch.Tensor | None = None,
    target_class: int = 1,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compute symptom-level feature attributions for clinical inputs.

    Measures marginal change in target class logit when each feature is perturbed,
    identifying symptoms that most strongly support or oppose the prediction.
    """
    model.eval()
    clin = clinical_tensor.clone().detach()
    if clin.ndim == 1:
        clin = clin.unsqueeze(0)

    device = next(model.parameters()).device
    clin = clin.to(device)

    # Base forward
    with torch.no_grad():
        if image_tensor is not None:
            img = image_tensor.to(device)
            if img.ndim == 3:
                img = img.unsqueeze(0)
            base_logits = model(img, clin)
        else:
            base_logits = model(clin)
        base_val = float(base_logits[0, target_class].item())

    # Perturbation attribution
    attributions = []
    num_features = clin.shape[1]

    for i in range(num_features):
        perturbed_clin = clin.clone()
        # Toggle feature (or zero out)
        perturbed_clin[0, i] = 0.0
        with torch.no_grad():
            if image_tensor is not None:
                perturbed_logits = model(img, perturbed_clin)
            else:
                perturbed_logits = model(perturbed_clin)
            perturbed_val = float(perturbed_logits[0, target_class].item())

        impact = base_val - perturbed_val
        name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        attributions.append(
            {
                "feature": name,
                "importance": float(impact),
                "direction": "supports" if impact > 0 else "opposes",
            }
        )

    # Rank by absolute impact
    ranked = sorted(attributions, key=lambda x: abs(x["importance"]), reverse=True)
    top_supporting = [item for item in ranked if item["importance"] > 0][:top_k]
    top_opposing = [item for item in ranked if item["importance"] < 0][:top_k]

    return {
        "attributions": ranked[: top_k * 2],
        "top_supporting_symptoms": top_supporting,
        "top_opposing_symptoms": top_opposing,
    }
