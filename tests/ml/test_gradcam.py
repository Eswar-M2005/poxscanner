"""Grad-CAM unit tests — tiny conv net, no dataset required."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.src.gradcam import (  # noqa: E402
    GradCAM,
    cam_to_png_bytes,
    heatmap_to_png_bytes,
    overlay_cam_on_image,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class TinyConvNet(nn.Module):
    """Minimal conv classifier so Grad-CAM has a spatial target layer."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def _random_input(height: int = 16, width: int = 16) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(1, 3, height, width)


def test_generate_shape_and_range() -> None:
    model = TinyConvNet()
    x = _random_input(16, 20)
    with GradCAM(model, model.conv) as engine:
        cam = engine.generate(x)

    assert isinstance(cam, np.ndarray)
    assert cam.ndim == 2
    assert cam.shape == (16, 20)
    assert np.isfinite(cam).all()
    assert float(cam.min()) >= 0.0
    assert float(cam.max()) <= 1.0


def test_generate_uses_argmax_when_class_idx_none() -> None:
    model = TinyConvNet()
    x = _random_input()
    with torch.no_grad():
        logits = model(x)
        expected = int(torch.argmax(logits, dim=1).item())

    with GradCAM(model, model.conv) as engine:
        cam_default = engine.generate(x, class_idx=None)
        cam_explicit = engine.generate(x, class_idx=expected)

    np.testing.assert_allclose(cam_default, cam_explicit, rtol=1e-5, atol=1e-5)


def test_generate_accepts_explicit_class_idx() -> None:
    model = TinyConvNet()
    x = _random_input()
    with GradCAM(model, model.conv) as engine:
        cam0 = engine.generate(x, class_idx=0)
        cam1 = engine.generate(x, class_idx=1)

    assert cam0.shape == cam1.shape == (16, 16)
    assert 0.0 <= float(cam0.min()) <= float(cam0.max()) <= 1.0
    assert 0.0 <= float(cam1.min()) <= float(cam1.max()) <= 1.0


def test_overlay_matches_image_hw_and_is_uint8_rgb() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(40, 50, 3), dtype=np.uint8)
    cam = rng.random((16, 16)).astype(np.float32)
    overlay = overlay_cam_on_image(image, cam)

    assert overlay.shape == (40, 50, 3)
    assert overlay.shape[:2] == image.shape[:2]
    assert overlay.dtype == np.uint8
    assert overlay.ndim == 3


def test_cam_to_png_bytes_starts_with_png_signature() -> None:
    overlay = np.zeros((8, 12, 3), dtype=np.uint8)
    overlay[:, :4] = (255, 0, 0)
    data = cam_to_png_bytes(overlay)
    assert isinstance(data, (bytes, bytearray))
    assert data.startswith(PNG_SIGNATURE)


def test_heatmap_to_png_bytes_starts_with_png_signature() -> None:
    cam = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    data = heatmap_to_png_bytes(cam)
    assert data.startswith(PNG_SIGNATURE)


def test_close_removes_hooks() -> None:
    model = TinyConvNet()
    n_fwd = len(model.conv._forward_hooks)
    engine = GradCAM(model, model.conv)
    assert len(model.conv._forward_hooks) == n_fwd + 1
    engine.close()
    assert len(model.conv._forward_hooks) == n_fwd
    engine.close()  # idempotent
    assert len(model.conv._forward_hooks) == n_fwd


def test_hooks_capture_activations_and_gradients() -> None:
    model = TinyConvNet()
    x = _random_input()
    engine = GradCAM(model, model.conv)
    try:
        cam = engine.generate(x)
        assert engine.activations is not None
        assert engine.gradients is not None
        assert engine.activations.ndim == 4
        assert engine.gradients.shape == engine.activations.shape
        assert cam.shape == (x.shape[2], x.shape[3])
    finally:
        engine.close()
