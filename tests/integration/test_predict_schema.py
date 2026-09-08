"""Integration checks for POST /api/v1/predict success JSON (contracts.md)."""

from __future__ import annotations

DISPLAY_LABELS = {"Monkeypox", "Non-Monkeypox"}
DISCLAIMER = (
    "Research and educational use only. This tool does not provide a medical diagnosis."
)


def test_predict_schema_types_and_invariants(api_client, tiny_png_bytes: bytes) -> None:
    response = api_client.post(
        "/api/v1/predict",
        files={"image": ("tiny.png", tiny_png_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["prediction"] in DISPLAY_LABELS
    assert isinstance(body["confidence"], (int, float))
    assert 0.0 <= float(body["confidence"]) <= 1.0

    probs = body["probabilities"]
    assert set(probs) == {"monkeypox", "non_monkeypox"}
    total = float(probs["monkeypox"]) + float(probs["non_monkeypox"])
    assert abs(total - 1.0) < 1e-5

    assert isinstance(body["model"], str) and body["model"]
    assert body["version"] == "0.1.0"
    assert isinstance(body["demo_mode"], bool)
    assert body["disclaimer"] == DISCLAIMER

    explanation = body["explanation"]
    assert explanation["method"] == "gradcam"
    overlay = explanation["overlay_png_base64"]
    assert isinstance(overlay, str) and overlay
    assert explanation["note"] is None or isinstance(explanation["note"], str)


def test_model_info_schema(api_client) -> None:
    response = api_client.get("/api/v1/model-info")
    assert response.status_code == 200
    body = response.json()
    required = {
        "name",
        "version",
        "task",
        "classes",
        "input",
        "metrics",
        "selection_status",
        "demo_mode",
        "disclaimer",
    }
    assert required.issubset(body.keys())
    assert body["classes"] == ["non_monkeypox", "monkeypox"]
    assert body["task"] == "binary_skin_lesion_classification"
    assert body["disclaimer"] == DISCLAIMER
