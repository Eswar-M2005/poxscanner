"""Placeholder metrics must stay null until a real evaluation writes n_samples > 0.

Does not invent scores. Does not load checkpoints or images.
"""

from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_METRICS_DIR = _REPO / "ml" / "artifacts" / "metrics"

PLACEHOLDER_MODELS = (
    "efficientnet_b0_test.json",
    "densenet121_test.json",
    "resnet50_test.json",
    "mobilenet_v3_small_test.json",
)

NULL_FIELDS = (
    "accuracy",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
    "f1",
    "roc_auc",
    "pr_auc",
    "latency_ms_p50",
)


def _load(name: str) -> dict:
    path = _METRICS_DIR / name
    assert path.is_file(), f"missing {path}"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_placeholder_or_real_metrics_are_internally_consistent() -> None:
    for name in PLACEHOLDER_MODELS:
        payload = _load(name)
        n_samples = payload.get("n_samples")
        assert n_samples is None or isinstance(n_samples, int)
        if not n_samples:
            for field in NULL_FIELDS:
                assert payload.get(field) is None, f"{name}.{field} must be null while n_samples is 0"
            matrix = payload.get("confusion_matrix")
            assert matrix == [[None, None], [None, None]]
            notes = str(payload.get("notes") or "")
            assert notes, f"{name} placeholder must explain nulls in notes"
            continue
        for field in NULL_FIELDS:
            value = payload[field]
            assert isinstance(value, (int, float)), f"{name}.{field} must be numeric after evaluation"
        matrix = payload["confusion_matrix"]
        assert len(matrix) == 2 and len(matrix[0]) == 2 and len(matrix[1]) == 2
        for row in matrix:
            for cell in row:
                assert isinstance(cell, int)


def test_selection_json_not_fabricated() -> None:
    payload = _load("selection.json")
    status = payload.get("status")
    selected = payload.get("selected_model")
    ranking = payload.get("ranking") or []
    if status == "pending_training":
        assert selected is None
        assert payload.get("checkpoint") in (None, "")
        assert ranking == []
        return
    assert selected, "completed selection must name a model"
    assert isinstance(ranking, list)
