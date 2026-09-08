"""Pytest fixtures and import path setup for contract tests.

Force demo mode before any FastAPI import so a mid-training ``*_best.pt``
cannot load torch/MPS and compete with an in-flight trainer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["DEMO_MODE"] = "true"
os.environ["STORE_UPLOADS"] = "false"
os.environ["MODEL_CHECKPOINT"] = ""

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_ROOT = REPO_ROOT / "server"

if SERVER_ROOT.is_dir() and str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def tiny_png_path(tmp_path: Path) -> Path:
    from tests.fixtures.make_image import write_tiny_rgb_png

    return write_tiny_rgb_png(tmp_path / "tiny.png")


@pytest.fixture
def tiny_png_bytes(tiny_png_path: Path) -> bytes:
    return tiny_png_path.read_bytes()


def _load_fastapi_app():
    try:
        from app.main import app
    except Exception as exc:  # noqa: BLE001 — skip unless the app is importable
        pytest.skip(f"FastAPI app not importable: {exc}")
    return app


@pytest.fixture
def api_client():
    """Starlette/FastAPI TestClient, or skip if the server package is missing."""
    app = _load_fastapi_app()
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        try:
            from starlette.testclient import TestClient
        except ImportError as exc:
            pytest.skip(f"TestClient not available: {exc}")
    with TestClient(app) as client:
        yield client
