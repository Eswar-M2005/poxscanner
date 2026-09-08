"""Unit tests for clinical data preprocessing and leakage prevention."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.src.clinical import ClinicalPreprocessor


def test_clinical_preprocessor_removes_id_target_and_leakage():
    df = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5],
            "Status": [1, 1, 1, 0, 0],
            "characteristic symptoms of Monkeypox": [1, 1, 1, 0, 0],
            "symptoms compatible with monkeypox": [1, 1, 0, 0, 0],
            "fever": [1, 1, 0, 1, 0],
            "rash": [1, 0, 1, 0, 0],
            "headache": [0, 1, 1, 0, 1],
        }
    )
    pre = ClinicalPreprocessor(max_features=5)
    pre.fit(df, y=df["Status"])

    assert "ID" not in pre.selected_features_
    assert "Status" not in pre.selected_features_
    assert "characteristic symptoms of Monkeypox" not in pre.selected_features_
    assert "symptoms compatible with monkeypox" not in pre.selected_features_
    assert set(pre.selected_features_).issubset({"fever", "rash", "headache"})


def test_clinical_preprocessor_transform_imputes_missing():
    df = pd.DataFrame(
        {
            "fever": [1, 1, 0, 0],
            "rash": [1, 0, 1, 0],
            "headache": [0, 1, 1, 0],
            "Status": [1, 1, 0, 0],
        }
    )
    pre = ClinicalPreprocessor(max_features=3)
    pre.fit(df, y=df["Status"])

    # Test dictionary with missing keys
    partial_input = {"fever": 1}
    out = pre.transform(partial_input)
    assert out.shape == (1, 3)
    assert not np.isnan(out).any()


def test_clinical_preprocessor_save_load(tmp_path):
    df = pd.DataFrame(
        {
            "fever": [1, 1, 0, 0],
            "rash": [1, 0, 1, 0],
            "Status": [1, 1, 0, 0],
        }
    )
    pre = ClinicalPreprocessor(max_features=2)
    pre.fit(df, y=df["Status"])
    save_path = tmp_path / "clinical_test.json"
    pre.save(save_path)

    loaded = ClinicalPreprocessor.load(save_path)
    assert loaded.is_fitted_
    assert loaded.selected_features_ == pre.selected_features_
    t1 = pre.transform(df)
    t2 = loaded.transform(df)
    np.testing.assert_allclose(t1, t2)
