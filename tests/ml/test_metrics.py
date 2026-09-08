"""Unit tests for binary_metrics and save_curves."""

from __future__ import annotations

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.src.metrics import binary_metrics, save_curves


def test_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = binary_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["sensitivity"] == metrics["recall"]
    assert metrics["specificity"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)


def test_known_confusion_matrix_matches_sklearn():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_prob = np.array([0.1, 0.7, 0.8, 0.2, 0.9, 0.3])
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = binary_metrics(y_true, y_prob)

    assert metrics["confusion_matrix"] == [[2, 1], [1, 2]]
    assert metrics["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert metrics["precision"] == pytest.approx(
        precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    )
    assert metrics["recall"] == pytest.approx(
        recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    )
    assert metrics["sensitivity"] == metrics["recall"]
    assert metrics["specificity"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(y_true, y_prob))
    assert metrics["pr_auc"] == pytest.approx(average_precision_score(y_true, y_prob))


def test_two_column_probabilities_use_positive_class():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.05, 0.95],
        ]
    )
    metrics = binary_metrics(y_true, y_prob)
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_all_negative_specificity_and_null_roc():
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.05])
    metrics = binary_metrics(y_true, y_prob)
    assert metrics["confusion_matrix"] == [[3, 1], [0, 0]]
    assert metrics["recall"] == pytest.approx(0.0)
    assert metrics["sensitivity"] == pytest.approx(0.0)
    assert metrics["specificity"] == pytest.approx(0.75)
    assert metrics["roc_auc"] is None


def test_save_curves_writes_plots(tmp_path):
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    written = save_curves(
        y_true=y_true,
        y_prob=y_prob,
        history={"train_loss": [1.0, 0.5], "val_loss": [1.1, 0.6]},
        out_dir=tmp_path,
        prefix="unit",
    )
    assert written["roc"].is_file()
    assert written["pr"].is_file()
    assert written["confusion_matrix"].is_file()
    assert written["loss"].is_file()
    for path in written.values():
        assert path.stat().st_size > 0
