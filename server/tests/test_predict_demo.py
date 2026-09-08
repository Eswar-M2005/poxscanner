"""Demo-mode predict tests (tiny PNG fixture, no trained model)."""

from __future__ import annotations

import base64

import pytest

from app.schemas.predict import DISCLAIMER

REQUIRED_KEYS = {
    "prediction",
    "confidence",
    "probabilities",
    "model",
    "version",
    "demo_mode",
    "disclaimer",
    "explanation",
}


def test_predict_demo_tiny_png(client, tiny_png) -> None:
    response = client.post(
        "/api/v1/predict",
        files={"image": ("lesion.png", tiny_png, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert REQUIRED_KEYS <= set(data.keys())
    assert data["prediction"] in {"Monkeypox", "Non-Monkeypox"}
    assert data["demo_mode"] is True
    assert data["disclaimer"] == DISCLAIMER
    probs = data["probabilities"]
    assert set(probs.keys()) == {"monkeypox", "non_monkeypox"}
    assert probs["monkeypox"] + probs["non_monkeypox"] == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= data["confidence"] <= 1.0
    predicted = data["prediction"]
    if predicted == "Monkeypox":
        assert data["confidence"] == probs["monkeypox"]
    else:
        assert data["confidence"] == probs["non_monkeypox"]
    explanation = data["explanation"]
    assert explanation is not None
    assert explanation["method"] == "gradcam"
    overlay = base64.b64decode(explanation["overlay_png_base64"])
    assert overlay[:8] == b"\x89PNG\r\n\x1a\n"
    assert explanation["note"]


def test_predict_demo_probabilities_sum_to_one(client, tiny_png) -> None:
    response = client.post(
        "/api/v1/predict",
        files={"image": ("lesion.png", tiny_png, "image/png")},
    )
    assert response.status_code == 200
    probs = response.json()["probabilities"]
    assert abs(probs["monkeypox"] + probs["non_monkeypox"] - 1.0) < 1e-6


def test_predict_demo_deterministic(client, tiny_png) -> None:
    files = {"image": ("lesion.png", tiny_png, "image/png")}
    first = client.post("/api/v1/predict", files=files).json()
    second = client.post("/api/v1/predict", files=files).json()
    assert first["probabilities"] == second["probabilities"]
    assert first["prediction"] == second["prediction"]


def test_predict_demo_without_explain_omits_overlay(client, tiny_png) -> None:
    response = client.post(
        "/api/v1/predict?explain=false",
        files={"image": ("lesion.png", tiny_png, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["explanation"] is None
