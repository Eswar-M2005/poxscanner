"""Model-info endpoint tests."""

from __future__ import annotations

from app.schemas.predict import DISCLAIMER


def test_model_info_demo(client) -> None:
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["name"]
    assert data["checkpoint"].endswith("ml/artifacts/checkpoints/selected.pt")
    assert data["task"] == "binary_skin_lesion_classification"
    assert data["classes"] == ["non_monkeypox", "monkeypox"]
    assert data["class_mapping"] == {"non_monkeypox": 0, "monkeypox": 1}
    assert data["input"]["image_size"] == 224
    assert data["input"]["formats"] == ["image/jpeg", "image/png", "image/webp"]
    assert data["input"]["normalization"] == {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    assert data["threshold"] == 0.15
    assert data["metrics"]["split"] == "test"
    assert data["metrics"]["n_samples"] == 270
    assert data["selection_status"] == "selected"
    assert data["demo_mode"] is True
    assert data["disclaimer"] == DISCLAIMER
    assert isinstance(data["version"], str)
