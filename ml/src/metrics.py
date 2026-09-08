"""Binary classification metrics and curve plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

VERSION = "0.1.0"

_SKLEARN_ERROR = (
    "scikit-learn is required for ml.src.metrics. "
    "Install it with: pip install scikit-learn"
)

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix as sk_confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
except ImportError:  # pragma: no cover
    accuracy_score = None  # type: ignore[assignment]


def _require_sklearn() -> None:
    if accuracy_score is None:
        raise RuntimeError(_SKLEARN_ERROR)


def _as_1d_prob(y_prob: np.ndarray) -> np.ndarray:
    arr = np.asarray(y_prob, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[1] < 2:
            raise ValueError("y_prob with shape (N, C) needs C >= 2")
        return arr[:, 1]
    return arr.reshape(-1)


def binary_metrics(
    y_true: Sequence | np.ndarray,
    y_prob: Sequence | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Positive class is monkeypox (index 1).

    Returns accuracy, precision, recall, sensitivity (=recall), specificity,
    f1, roc_auc, pr_auc, and confusion_matrix ``[[TN, FP], [FN, TP]]``.
    """
    _require_sklearn()
    y_true_arr = np.asarray(y_true).reshape(-1).astype(int)
    y_prob_arr = _as_1d_prob(np.asarray(y_prob))
    if y_true_arr.shape[0] != y_prob_arr.shape[0]:
        raise ValueError("y_true and y_prob must have the same length")
    if y_true_arr.size == 0:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "sensitivity": None,
            "specificity": None,
            "f1": None,
            "roc_auc": None,
            "pr_auc": None,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    y_pred = (y_prob_arr >= threshold).astype(int)
    cm = sk_confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    recall = float(recall_score(y_true_arr, y_pred, pos_label=1, zero_division=0))
    spec_denom = tn + fp
    specificity = float(tn / spec_denom) if spec_denom else 0.0

    roc_auc: float | None
    pr_auc: float | None
    try:
        if len(np.unique(y_true_arr)) < 2:
            roc_auc = None
        else:
            roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        roc_auc = None
    try:
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    except ValueError:
        pr_auc = None

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(
            precision_score(y_true_arr, y_pred, pos_label=1, zero_division=0)
        ),
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": float(f1_score(y_true_arr, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def metrics_payload(
    model: str,
    split: str,
    metrics: dict[str, Any] | None,
    n_samples: int = 0,
    notes: str = "",
    latency_ms_p50: float | None = None,
) -> dict[str, Any]:
    """Build the metrics JSON object from contracts.md."""
    keys = (
        "accuracy",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
    )
    row: dict[str, Any] = {k: None for k in keys}
    cm: list[list[Any]] = [[None, None], [None, None]]
    if metrics:
        for key in keys:
            value = metrics.get(key)
            row[key] = None if value is None else float(value)
        raw_cm = metrics.get("confusion_matrix")
        if raw_cm is not None:
            cm = [[int(v) for v in raw_cm[0]], [int(v) for v in raw_cm[1]]]
    return {
        "model": model,
        "split": split,
        "version": VERSION,
        "n_samples": int(n_samples),
        **row,
        "latency_ms_p50": None if latency_ms_p50 is None else float(latency_ms_p50),
        "confusion_matrix": cm,
        "notes": notes,
    }


def dump_metrics_json(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _plots_dir(out_dir: Path | str | None) -> Path:
    if out_dir is not None:
        return Path(out_dir)
    try:
        from ml.src.utils import repo_root

        return repo_root() / "ml" / "artifacts" / "plots"
    except Exception:
        return Path("ml/artifacts/plots")


def save_curves(
    y_true: Iterable | np.ndarray | None = None,
    y_prob: Iterable | np.ndarray | None = None,
    confusion_matrix: Sequence | np.ndarray | None = None,
    history: dict[str, Sequence[float]] | None = None,
    out_dir: Path | str | None = None,
    prefix: str = "",
) -> dict[str, Path]:
    """Write ROC, PR, confusion-matrix, and loss plots under ``out_dir``."""
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dest = _plots_dir(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_" if prefix else ""
    written: dict[str, Path] = {}

    y_true_arr = None if y_true is None else np.asarray(y_true).reshape(-1).astype(int)
    y_prob_arr = None if y_prob is None else _as_1d_prob(np.asarray(y_prob))

    if y_true_arr is not None and y_prob_arr is not None and y_true_arr.size:
        _require_sklearn()
        fig, ax = plt.subplots(figsize=(5, 5))
        if len(np.unique(y_true_arr)) >= 2:
            fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
            try:
                auc_val = roc_auc_score(y_true_arr, y_prob_arr)
                label = f"ROC-AUC = {auc_val:.3f}"
            except ValueError:
                label = "ROC"
            ax.plot(fpr, tpr, color="#0F766E", label=label)
        ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", linewidth=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve")
        ax.legend(loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        roc_path = dest / f"{stem}roc.png"
        fig.savefig(roc_path, dpi=120)
        plt.close(fig)
        written["roc"] = roc_path

        fig, ax = plt.subplots(figsize=(5, 5))
        prec, rec, _ = precision_recall_curve(y_true_arr, y_prob_arr)
        try:
            ap = average_precision_score(y_true_arr, y_prob_arr)
            label = f"PR-AUC = {ap:.3f}"
        except ValueError:
            label = "PR"
        ax.plot(rec, prec, color="#102A43", label=label)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision–recall curve")
        ax.legend(loc="lower left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        pr_path = dest / f"{stem}pr.png"
        fig.savefig(pr_path, dpi=120)
        plt.close(fig)
        written["pr"] = pr_path

    cm = confusion_matrix
    if cm is None and y_true_arr is not None and y_prob_arr is not None and y_true_arr.size:
        _require_sklearn()
        cm = binary_metrics(y_true_arr, y_prob_arr)["confusion_matrix"]
    if cm is not None:
        cm_arr = np.asarray(cm, dtype=float)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(cm_arr, cmap="Blues")
        ax.set_xticks([0, 1], labels=["Non-Monkeypox", "Monkeypox"])
        ax.set_yticks([0, 1], labels=["Non-Monkeypox", "Monkeypox"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion matrix")
        for (i, j), value in np.ndenumerate(cm_arr):
            ax.text(j, i, f"{int(value)}", ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        cm_path = dest / f"{stem}confusion_matrix.png"
        fig.savefig(cm_path, dpi=120)
        plt.close(fig)
        written["confusion_matrix"] = cm_path

    if history:
        fig, ax = plt.subplots(figsize=(6, 4))
        if "train_loss" in history:
            ax.plot(list(history["train_loss"]), label="Train loss", color="#102A43")
        if "val_loss" in history:
            ax.plot(list(history["val_loss"]), label="Val loss", color="#0F766E")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss curves")
        ax.legend()
        fig.tight_layout()
        loss_path = dest / f"{stem}loss.png"
        fig.savefig(loss_path, dpi=120)
        plt.close(fig)
        written["loss"] = loss_path

    return written
