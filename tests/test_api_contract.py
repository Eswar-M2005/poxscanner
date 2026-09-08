"""HTTP contract: /health and /api/v1/predict JSON keys from docs/contracts.md."""

from __future__ import annotations

HEALTH_KEYS = {"status", "service", "model_loaded", "demo_mode", "version"}
PREDICT_KEYS = {
    "prediction",
    "confidence",
    "probabilities",
    "model",
    "version",
    "demo_mode",
    "disclaimer",
    "explanation",
}


def test_health_keys(api_client) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert HEALTH_KEYS.issubset(body.keys())
    assert body["service"] == "poxscanner-api"
    assert body["status"] == "ok"
    assert isinstance(body["model_loaded"], bool)
    assert isinstance(body["demo_mode"], bool)


def test_predict_tiny_image_keys(api_client, tiny_png_bytes: bytes) -> None:
    response = api_client.post(
        "/api/v1/predict",
        files={"image": ("tiny.png", tiny_png_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert PREDICT_KEYS.issubset(body.keys())
    assert "monkeypox" in body["probabilities"]
    assert "non_monkeypox" in body["probabilities"]
    assert isinstance(body["explanation"], dict)
    assert {"method", "overlay_png_base64", "note"}.issubset(body["explanation"].keys())
