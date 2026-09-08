"""Authoritative model artifact metadata loader."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import settings

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SERVER_ROOT = Path(__file__).resolve().parents[2]

_MANIFEST_PATH = _REPO_ROOT / "ml" / "artifacts" / "metrics" / "model_manifest.json"
_SELECTION_PATH = _REPO_ROOT / "ml" / "artifacts" / "metrics" / "selection.json"
_THRESHOLD_PATH = (
    _REPO_ROOT / "ml" / "artifacts" / "metrics" / "threshold_analysis.json"
)
_DEFAULT_CHECKPOINT = _REPO_ROOT / "ml" / "artifacts" / "checkpoints" / "selected.pt"
_DEFAULT_TASK = "binary_skin_lesion_classification"
_DEFAULT_CLASSES = {"non_monkeypox": 0, "monkeypox": 1}
_DEFAULT_FORMATS = ["image/jpeg", "image/png", "image/webp"]
_DEFAULT_IMAGE_SIZE = 224
_DEFAULT_THRESHOLD = 0.5
_DEFAULT_NORMALIZATION = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def read_json(path: Path) -> dict[str, Any] | None:
    return _read_json(path)


def _coerce_threshold(value: Any) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return _DEFAULT_THRESHOLD
    return threshold if 0.0 <= threshold <= 1.0 else _DEFAULT_THRESHOLD


def _read_normalization() -> dict[str, list[float]]:
    return {
        "mean": list(_DEFAULT_NORMALIZATION["mean"]),
        "std": list(_DEFAULT_NORMALIZATION["std"]),
    }


@lru_cache
def manifest() -> dict[str, Any]:
    return _read_json(_MANIFEST_PATH) or {}


@lru_cache
def selection() -> dict[str, Any]:
    return _read_json(_SELECTION_PATH) or {}


@lru_cache
def threshold_analysis() -> dict[str, Any]:
    return _read_json(_THRESHOLD_PATH) or {}


def model_name() -> str:
    data = manifest()
    for key in ("name", "selected_model"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = selection().get("selected_model")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "resnet50"


def checkpoint_path() -> Path | None:
    raw = (settings.MODEL_CHECKPOINT or "").strip()
    if raw:
        candidate = Path(raw)
        if candidate.is_file():
            return candidate.resolve()
        for base in (_SERVER_ROOT, _REPO_ROOT):
            nested = (base / raw).resolve()
            if nested.is_file():
                return nested
        return None

    candidate = manifest().get("checkpoint")
    if isinstance(candidate, str) and candidate.strip():
        path = Path(candidate)
        if path.is_file():
            return path.resolve()
        resolved = (_REPO_ROOT / candidate).resolve()
        if resolved.is_file():
            return resolved

    candidate = selection().get("checkpoint")
    if isinstance(candidate, str) and candidate.strip():
        path = Path(candidate)
        if path.is_file():
            return path.resolve()
        resolved = (_REPO_ROOT / candidate).resolve()
        if resolved.is_file():
            return resolved

    if _DEFAULT_CHECKPOINT.is_file():
        return _DEFAULT_CHECKPOINT.resolve()
    return None


def selection_status() -> str:
    value = manifest().get("selection_status")
    if isinstance(value, str) and value.strip():
        return value.strip()
    value = selection().get("status")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "pending_training"


def version() -> str:
    value = manifest().get("version")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return settings.APP_VERSION


def class_mapping() -> dict[str, int]:
    raw = manifest().get("class_mapping")
    if isinstance(raw, dict) and raw:
        mapping: dict[str, int] = {}
        for key, value in raw.items():
            try:
                mapping[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        if mapping:
            return mapping
    return dict(_DEFAULT_CLASSES)


def classes() -> list[str]:
    mapping = class_mapping()
    return sorted(mapping.keys(), key=lambda key: mapping[key])


def image_size() -> int:
    raw = manifest().get("image_size")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_IMAGE_SIZE
    return value if value > 0 else _DEFAULT_IMAGE_SIZE


def threshold() -> float:
    raw = threshold_analysis().get("recommended_threshold")
    if isinstance(raw, dict) and "threshold" in raw:
        return _coerce_threshold(raw.get("threshold"))
    raw = manifest().get("threshold")
    if raw is not None:
        return _coerce_threshold(raw)
    return _DEFAULT_THRESHOLD


def normalization() -> dict[str, list[float]]:
    return _read_normalization()


def metrics() -> dict[str, Any] | None:
    raw = manifest().get("metrics")
    if isinstance(raw, dict) and raw:
        return raw
    return None


def disclaimer() -> str:
    value = manifest().get("disclaimer")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "Research and educational use only. This tool does not provide a medical diagnosis."


def input_formats() -> list[str]:
    return list(_DEFAULT_FORMATS)


def model_info_payload(*, demo_mode: bool) -> dict[str, Any]:
    return {
        "name": model_name(),
        "version": version(),
        "task": _DEFAULT_TASK,
        "classes": classes(),
        "class_mapping": class_mapping(),
        "checkpoint": str(checkpoint_path() or ""),
        "input": {
            "image_size": image_size(),
            "formats": input_formats(),
            "normalization": normalization(),
        },
        "threshold": threshold(),
        "metrics": metrics(),
        "selection_status": selection_status(),
        "demo_mode": demo_mode,
        "disclaimer": disclaimer(),
    }
