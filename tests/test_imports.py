"""Smoke-import shared utils; skip optional heavy modules when absent."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pytest

OPTIONAL_MODULES = (
    "torch",
    "torchvision",
    "fastapi",
    "app.main",
    "ml.src.models",
    "ml.src.train",
    "ml.src.evaluate",
    "ml.src.predict",
    "ml.src.metrics",
    "ml.src.dataset",
    "ml.src.transforms",
    "ml.src.gradcam",
    "ml.src.audit",
    "ml.src.architectures.efficientnet",
)


def test_utils_importable() -> None:
    mod = importlib.import_module("ml.src.utils")
    for name in ("repo_root", "set_seed", "load_yaml", "sha256_file", "get_logger"):
        assert callable(getattr(mod, name))


def test_utils_repo_root_hash_logger(tmp_path: Path) -> None:
    from ml.src.utils import get_logger, repo_root, sha256_file

    root = repo_root()
    assert (root / "docs" / "contracts.md").is_file()
    blob = tmp_path / "probe.bin"
    blob.write_bytes(b"poxscanner")
    assert sha256_file(blob) == hashlib.sha256(b"poxscanner").hexdigest()
    logger = get_logger("poxscanner.test")
    assert logger.name == "poxscanner.test"


def test_load_yaml_mapping(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    from ml.src.utils import load_yaml

    path = tmp_path / "sample.yaml"
    path.write_text("model:\n  name: efficientnet_b0\n", encoding="utf-8")
    assert load_yaml(path)["model"]["name"] == "efficientnet_b0"


@pytest.mark.parametrize("module_name", OPTIONAL_MODULES)
def test_optional_heavy_modules(module_name: str) -> None:
    pytest.importorskip(module_name)
