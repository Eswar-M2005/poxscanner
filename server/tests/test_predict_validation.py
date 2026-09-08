"""Predict input validation tests (no checkpoint / GPU required)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services import inference as inference_module


def test_predict_bad_mime(client, tiny_png) -> None:
    response = client.post(
        "/api/v1/predict",
        files={"image": ("notes.txt", tiny_png, "text/plain")},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "unsupported_media_type"
    assert "message" in body["error"]


def test_predict_too_large(client, monkeypatch) -> None:
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 64)
    payload = b"\x89PNG\r\n\x1a\n" + b"x" * 200
    response = client.post(
        "/api/v1/predict",
        files={"image": ("big.png", payload, "image/png")},
    )
    assert response.status_code == 413
    body = response.json()
    assert body["error"]["code"] == "file_too_large"
    assert "message" in body["error"]


def test_predict_corrupt_bytes(client) -> None:
    response = client.post(
        "/api/v1/predict",
        files={"image": ("bad.png", b"this is not an image", "image/png")},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "invalid_image"
    assert body["error"]["message"] == "File could not be decoded as an image."


def test_predict_503_when_demo_mode_disabled(monkeypatch, tiny_png) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", "false")
    monkeypatch.setattr(
        "app.services.inference.model_registry.checkpoint_path",
        lambda: None,
    )
    previous_service = inference_module._SERVICE
    inference_module._SERVICE = None
    try:
        with TestClient(app) as test_client:
            response = test_client.post(
                "/api/v1/predict",
                files={"image": ("lesion.png", tiny_png, "image/png")},
            )
    finally:
        inference_module._SERVICE = previous_service

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "model_unavailable"
