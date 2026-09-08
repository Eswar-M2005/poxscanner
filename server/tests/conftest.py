"""Pytest fixtures. Force demo mode so tests never load torch."""

from __future__ import annotations

import os
import sys
from io import BytesIO
from pathlib import Path

os.environ["DEMO_MODE"] = "true"
os.environ["STORE_UPLOADS"] = "false"
os.environ["MODEL_CHECKPOINT"] = ""

SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def tiny_png() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (8, 8), color=(120, 40, 30)).save(buf, format="PNG")
    return buf.getvalue()
