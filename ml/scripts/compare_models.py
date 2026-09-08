"""Rank completed experiments and write selection.json.

CLI::

    python -m ml.scripts.compare_models
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml.src.utils import get_logger, repo_root

PENDING = {
    "status": "pending_training",
    "selected_model": None,
    "checkpoint": None,
    "reason": "No completed experiment metrics. Add images under data/raw and run training.",
    "ranking": [],
}

log = get_logger("ml.scripts.compare_models")


def _is_completed(payload: dict[str, Any]) -> bool:
    return payload.get("accuracy") is not None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _float_or(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rank_key(row: dict[str, Any]) -> tuple:
    recall = _float_or(row.get("recall"), _float_or(row.get("sensitivity"), float("-inf")))
    f1 = _float_or(row.get("f1"), float("-inf"))
    roc = _float_or(row.get("roc_auc"), float("-inf"))
    pr = _float_or(row.get("pr_auc"), float("-inf"))
    spec = _float_or(row.get("specificity"), float("-inf"))
    gap = _float_or(row.get("f1_val_test_gap"), 0.0)
    latency = _float_or(row.get("latency_ms_p50"), float("inf"))
    return (-recall, -f1, -roc, -pr, -spec, gap, latency)


def _ranking_entry(test: dict[str, Any], val: dict[str, Any] | None) -> dict[str, Any]:
    recall = test.get("recall") if test.get("recall") is not None else test.get("sensitivity")
    val_f1 = None if val is None else val.get("f1")
    test_f1 = test.get("f1")
    gap = None
    if val_f1 is not None and test_f1 is not None:
        gap = abs(float(val_f1) - float(test_f1))
    return {
        "model": test.get("model"),
        "recall": recall,
        "sensitivity": test.get("sensitivity", recall),
        "f1": test_f1,
        "roc_auc": test.get("roc_auc"),
        "pr_auc": test.get("pr_auc"),
        "specificity": test.get("specificity"),
        "accuracy": test.get("accuracy"),
        "latency_ms_p50": test.get("latency_ms_p50"),
        "f1_val_test_gap": gap,
        "n_samples": test.get("n_samples"),
    }


def compare_and_select(metrics_dir: Path, checkpoint_dir: Path | None = None) -> dict[str, Any]:
    test_files = sorted(metrics_dir.glob("*_test.json"))
    completed: list[dict[str, Any]] = []
    for path in test_files:
        payload = _load_json(path)
        if payload is None or not _is_completed(payload):
            continue
        model_name = payload.get("model") or path.name.removesuffix("_test.json")
        val_path = metrics_dir / f"{model_name}_val.json"
        val_payload = _load_json(val_path) if val_path.is_file() else None
        if val_payload is not None and not _is_completed(val_payload):
            val_payload = None
        entry = _ranking_entry(payload, val_payload)
        entry["metrics_file"] = str(path.as_posix())
        completed.append(entry)

    if not completed:
        return dict(PENDING)

    ranking = sorted(completed, key=_rank_key)
    winner = ranking[0]
    name = winner["model"]
    ckpt = None
    if name:
        rel = f"ml/artifacts/checkpoints/{name}_best.pt"
        if checkpoint_dir is not None:
            candidate = checkpoint_dir / f"{name}_best.pt"
            ckpt = str(candidate.as_posix()) if candidate.is_file() else rel
        else:
            ckpt = rel
    return {
        "status": "selected",
        "selected_model": name,
        "checkpoint": ckpt,
        "reason": (
            "Ranked by monkeypox recall/sensitivity, then F1, ROC-AUC, PR-AUC, "
            "specificity, |val−test| F1 gap, and inference latency."
        ),
        "ranking": ranking,
    }


def _copy_selected_checkpoint(root: Path, result: dict[str, Any]) -> None:
    if result.get("status") != "selected" or not result.get("selected_model"):
        return
    name = str(result["selected_model"])
    src = root / "ml" / "artifacts" / "checkpoints" / f"{name}_best.pt"
    dest = root / "ml" / "artifacts" / "checkpoints" / "selected.pt"
    if not src.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    shutil.copy2(src, dest)
    result["checkpoint"] = "ml/artifacts/checkpoints/selected.pt"


def write_model_manifest(root: Path, result: dict[str, Any]) -> Path | None:
    metrics_dir = root / "ml" / "artifacts" / "metrics"
    dest = metrics_dir / "model_manifest.json"
    if result.get("status") != "selected" or not result.get("selected_model"):
        return None
    name = str(result["selected_model"])
    test_metrics = _load_json(metrics_dir / f"{name}_test.json") or {}
    payload = {
        "name": name,
        "selected_model": name,
        "checkpoint": result.get("checkpoint") or f"ml/artifacts/checkpoints/{name}_best.pt",
        "version": "0.1.0",
        "class_mapping": {"non_monkeypox": 0, "monkeypox": 1},
        "image_size": 224,
        "metrics": {
            "split": test_metrics.get("split", "test"),
            "n_samples": test_metrics.get("n_samples"),
            "accuracy": test_metrics.get("accuracy"),
            "precision": test_metrics.get("precision"),
            "recall": test_metrics.get("recall"),
            "sensitivity": test_metrics.get("sensitivity"),
            "specificity": test_metrics.get("specificity"),
            "f1": test_metrics.get("f1"),
            "roc_auc": test_metrics.get("roc_auc"),
            "pr_auc": test_metrics.get("pr_auc"),
            "latency_ms_p50": test_metrics.get("latency_ms_p50"),
            "confusion_matrix": test_metrics.get("confusion_matrix"),
        },
        "selection_status": result.get("status"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": (
            "Research and educational use only. This tool does not provide a medical diagnosis."
        ),
    }
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest


def main(argv: list[str] | None = None) -> int:
    del argv
    root = repo_root()
    metrics_dir = root / "ml" / "artifacts" / "metrics"
    checkpoint_dir = root / "ml" / "artifacts" / "checkpoints"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    result = compare_and_select(metrics_dir, checkpoint_dir=checkpoint_dir)
    _copy_selected_checkpoint(root, result)
    dest = metrics_dir / "selection.json"
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    manifest = write_model_manifest(root, result)
    log.info("Wrote %s (status=%s)", dest, result.get("status"))
    if manifest is not None:
        log.info("Wrote %s", manifest)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
