"""Threshold-aware prediction tests."""

from __future__ import annotations

import sys
import types

from PIL import Image

from app.services.inference import InferenceService


def _image() -> Image.Image:
    return Image.new("RGB", (32, 32), color=(128, 80, 32))


def test_real_prediction_uses_manifest_threshold(monkeypatch) -> None:
    service = InferenceService()
    service._model_loaded = True
    service._model = object()
    service._meta = {"model_name": "resnet50", "version": "0.1.0", "image_size": 224}

    def fake_predict_image(_model, _image, _meta):
        return {
            "probabilities": {
                "monkeypox": 0.2,
                "non_monkeypox": 0.8,
            }
        }

    fake_module = types.SimpleNamespace(predict_image=fake_predict_image)
    monkeypatch.setitem(sys.modules, "ml.src.predict", fake_module)
    monkeypatch.setattr(service, "_deployment_threshold", lambda: 0.15)
    result = service._predict_real(b"demo", _image(), explain=False)
    assert result["prediction"] == "Monkeypox"
    assert result["confidence"] == 0.2


def test_real_prediction_below_threshold(monkeypatch) -> None:
    service = InferenceService()
    service._model_loaded = True
    service._model = object()
    service._meta = {"model_name": "resnet50", "version": "0.1.0", "image_size": 224}

    def fake_predict_image(_model, _image, _meta):
        return {
            "probabilities": {
                "monkeypox": 0.1,
                "non_monkeypox": 0.9,
            }
        }

    fake_module = types.SimpleNamespace(predict_image=fake_predict_image)
    monkeypatch.setitem(sys.modules, "ml.src.predict", fake_module)
    monkeypatch.setattr(service, "_deployment_threshold", lambda: 0.15)
    result = service._predict_real(b"demo", _image(), explain=False)
    assert result["prediction"] == "Non-Monkeypox"
    assert result["confidence"] == 0.9
