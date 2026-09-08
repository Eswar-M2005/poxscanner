"""Integration tests for multimodal prediction API endpoint."""

from __future__ import annotations

import io
import json
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _make_dummy_jpeg() -> bytes:
    img = Image.new("RGB", (224, 224), color=(180, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_predict_image_only():
    img_bytes = _make_dummy_jpeg()
    with TestClient(app) as c:
        response = c.post(
            "/api/v1/predict",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            params={"explain": "true"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in ("Monkeypox", "Non-Monkeypox")
        assert "probabilities" in data
        assert data["modality"] == "image_only"
        assert "Model A" in data["model"]
        assert data["multimodal"] is False
        assert data["explanation"] is not None
        assert data["clinical_explanation"] is None


def test_predict_clinical_only():
    symptoms = {"fever": 1, "rash": 1, "oral and genital ulcers": 1}
    with TestClient(app) as c:
        response = c.post(
            "/api/v1/predict",
            data={"clinical_data": json.dumps(symptoms)},
            params={"explain": "true"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in ("Monkeypox", "Non-Monkeypox")
        assert "probabilities" in data
        assert data["modality"] == "clinical_only"
        assert "Model B" in data["model"]
        assert data["multimodal"] is False
        assert data["explanation"] is None
        assert data["clinical_explanation"] is not None
        assert "top_supporting_symptoms" in data["clinical_explanation"]


def test_predict_multimodal_with_both():
    img_bytes = _make_dummy_jpeg()
    symptoms = {"fever": 1, "rash": 1, "oral and genital ulcers": 1}
    with TestClient(app) as c:
        response = c.post(
            "/api/v1/predict",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"clinical_data": json.dumps(symptoms)},
            params={"explain": "true"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in ("Monkeypox", "Non-Monkeypox")
        assert "probabilities" in data
        assert data["modality"] == "multimodal"
        assert "Model C" in data["model"]
        assert data["multimodal"] is True
        assert data["explanation"] is not None
        assert data["clinical_explanation"] is not None
        assert "top_supporting_symptoms" in data["clinical_explanation"]


def test_predict_missing_both_inputs():
    with TestClient(app) as c:
        response = c.post(
            "/api/v1/predict",
        )
        assert response.status_code == 400
        data = response.json()
        assert data["error"]["code"] == "missing_input"
