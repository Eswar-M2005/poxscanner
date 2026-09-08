"""Health endpoint tests."""

from __future__ import annotations


def test_health_ok(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "poxscanner-api"
    assert data["model_loaded"] is False
    assert data["demo_mode"] is True
    assert isinstance(data["version"], str)
    assert data["version"]
