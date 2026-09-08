"""Shared path, seed, YAML, hashing, and logging helpers."""

from __future__ import annotations

import hashlib
import logging
import random
import sys
from pathlib import Path
from typing import Any

__all__ = [
    "repo_root",
    "set_seed",
    "load_yaml",
    "sha256_file",
    "get_logger",
]


def repo_root() -> Path:
    """Walk parents until ``docs/contracts.md`` or ``pyproject.toml`` exists."""
    starts = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    seen: set[Path] = set()
    for start in starts:
        for candidate in (start, *start.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "docs" / "contracts.md").is_file() or (
                candidate / "pyproject.toml"
            ).is_file():
                return candidate
    raise FileNotFoundError(
        "Could not locate repository root (docs/contracts.md or pyproject.toml)."
    )


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (if present). Make cuDNN deterministic."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping with PyYAML. Empty files become ``{}``."""
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def sha256_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with a stderr handler (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
