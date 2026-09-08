"""Clinical tabular data preprocessing and feature selection.

Handles:
- Identification and removal of identifiers ('ID') and target leakage columns.
- Collinearity and zero-variance filtering.
- Supervised feature selection fitted strictly on training data.
- Missing value imputation and normalization for inference.
- Serialization of preprocessor state to JSON for production inference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger("ml.src.clinical")

DEFAULT_IDENTIFIERS = {"id", "patient_id", "case_id", "record_id"}
DEFAULT_TARGET_COLS = {"status", "label", "target", "diagnosis"}
# Symptoms that explicitly name or directly encode the diagnosis
DEFAULT_LEAKAGE_COLS = {
    "characteristic symptoms of monkeypox",
    "symptoms compatible with monkeypox",
}


class ClinicalPreprocessor:
    """Preprocessor for clinical symptom features.

    Must be fitted ONLY on the training split to avoid data leakage.
    """

    def __init__(
        self,
        max_features: int = 32,
        correlation_threshold: float = 0.95,
        identifiers: Sequence[str] | None = None,
        target_cols: Sequence[str] | None = None,
        leakage_cols: Sequence[str] | None = None,
        random_state: int = 42,
    ) -> None:
        self.max_features = max_features
        self.correlation_threshold = correlation_threshold
        self.identifiers = {c.lower() for c in (identifiers or DEFAULT_IDENTIFIERS)}
        self.target_cols = {c.lower() for c in (target_cols or DEFAULT_TARGET_COLS)}
        self.leakage_cols = {c.lower() for c in (leakage_cols or DEFAULT_LEAKAGE_COLS)}
        self.random_state = random_state

        # Fitted state
        self.all_input_columns_: list[str] = []
        self.selected_features_: list[str] = []
        self.feature_means_: dict[str, float] = {}
        self.feature_stds_: dict[str, float] = {}
        self.feature_medians_: dict[str, float] = {}
        self.feature_importances_: dict[str, float] = {}
        self.is_fitted_: bool = False

    def _filter_candidate_columns(self, df: pd.DataFrame) -> list[str]:
        """Exclude ID, target, and known leakage columns."""
        candidates = []
        for col in df.columns:
            clean = str(col).strip().lower()
            if clean in self.identifiers:
                continue
            if clean in self.target_cols:
                continue
            if clean in self.leakage_cols:
                logger.info("Excluding leakage column: %s", col)
                continue
            candidates.append(col)
        return candidates

    def fit(self, df: pd.DataFrame, y: Sequence[int] | np.ndarray | None = None) -> "ClinicalPreprocessor":
        """Fit feature selection and scaling parameters on training data."""
        candidates = self._filter_candidate_columns(df)
        self.all_input_columns_ = candidates

        X = df[candidates].copy().astype(np.float32)

        # 1. Remove constant / zero-variance columns
        variances = X.var()
        non_constant = variances[variances > 1e-6].index.tolist()
        if not non_constant:
            non_constant = candidates
        X = X[non_constant]

        # 2. Remove redundant collinear columns
        corr = X.corr().abs()
        to_drop = set()
        cols = X.columns.tolist()
        for i in range(len(cols)):
            col_i = cols[i]
            if col_i in to_drop:
                continue
            for j in range(i + 1, len(cols)):
                col_j = cols[j]
                if col_j in to_drop:
                    continue
                if corr.loc[col_i, col_j] >= self.correlation_threshold:
                    to_drop.add(col_j)
                    logger.debug(
                        "Dropping collinear feature %s (corr %.2f with %s)",
                        col_j,
                        corr.loc[col_i, col_j],
                        col_i,
                    )

        remaining_cols = [c for c in cols if c not in to_drop]
        X = X[remaining_cols]

        # 3. Supervised feature ranking if labels provided
        if y is not None and len(remaining_cols) > self.max_features:
            y_arr = np.asarray(y, dtype=int)
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                class_weight="balanced",
                random_state=self.random_state,
            )
            rf.fit(X, y_arr)
            importances = dict(zip(remaining_cols, rf.feature_importances_))
            sorted_feats = sorted(importances.items(), key=lambda item: item[1], reverse=True)
            self.selected_features_ = [k for k, _ in sorted_feats[: self.max_features]]
            self.feature_importances_ = {k: float(v) for k, v in sorted_feats}
        else:
            self.selected_features_ = remaining_cols[: self.max_features]
            self.feature_importances_ = {k: 1.0 / len(self.selected_features_) for k in self.selected_features_}

        # 4. Compute statistics for normalization and missing-value imputation
        for feat in self.selected_features_:
            series = X[feat].dropna()
            self.feature_means_[feat] = float(series.mean()) if len(series) > 0 else 0.0
            std = float(series.std()) if len(series) > 0 else 1.0
            self.feature_stds_[feat] = std if std > 1e-4 else 1.0
            self.feature_medians_[feat] = float(series.median()) if len(series) > 0 else 0.0

        self.is_fitted_ = True
        logger.info(
            "Fitted ClinicalPreprocessor: %d initial candidate columns -> %d selected features",
            len(candidates),
            len(self.selected_features_),
        )
        return self

    def transform(self, data: pd.DataFrame | dict[str, Any] | Sequence[dict[str, Any]]) -> np.ndarray:
        """Transform clinical features into a normalized float32 array of shape (N, D)."""
        if not self.is_fitted_:
            raise RuntimeError("ClinicalPreprocessor must be fitted before calling transform()")

        if isinstance(data, dict):
            frame = pd.DataFrame([data])
        elif isinstance(data, (list, tuple)):
            frame = pd.DataFrame(list(data))
        elif isinstance(data, pd.DataFrame):
            frame = data.copy()
        else:
            raise TypeError(f"Unsupported data type for clinical transform: {type(data)}")

        rows = []
        for _, row in frame.iterrows():
            row_vals = []
            for feat in self.selected_features_:
                val = row.get(feat, None)
                if val is None or pd.isna(val):
                    val = self.feature_medians_.get(feat, 0.0)
                else:
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        val = self.feature_medians_.get(feat, 0.0)
                # Standardize
                mean = self.feature_means_.get(feat, 0.0)
                std = self.feature_stds_.get(feat, 1.0)
                norm_val = (val - mean) / std
                row_vals.append(norm_val)
            rows.append(row_vals)

        return np.array(rows, dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        """Serialize preprocessor state to a JSON-serializable dictionary."""
        return {
            "max_features": self.max_features,
            "correlation_threshold": self.correlation_threshold,
            "selected_features": self.selected_features_,
            "feature_means": self.feature_means_,
            "feature_stds": self.feature_stds_,
            "feature_medians": self.feature_medians_,
            "feature_importances": self.feature_importances_,
            "is_fitted": self.is_fitted_,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClinicalPreprocessor":
        """Reconstruct ClinicalPreprocessor from serialized dictionary."""
        obj = cls(
            max_features=data.get("max_features", 32),
            correlation_threshold=data.get("correlation_threshold", 0.95),
        )
        obj.selected_features_ = list(data.get("selected_features", []))
        obj.feature_means_ = {k: float(v) for k, v in data.get("feature_means", {}).items()}
        obj.feature_stds_ = {k: float(v) for k, v in data.get("feature_stds", {}).items()}
        obj.feature_medians_ = {k: float(v) for k, v in data.get("feature_medians", {}).items()}
        obj.feature_importances_ = {k: float(v) for k, v in data.get("feature_importances", {}).items()}
        obj.is_fitted_ = bool(data.get("is_fitted", False))
        return obj

    def save(self, path: Path | str) -> None:
        """Save preprocessor state to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "ClinicalPreprocessor":
        """Load preprocessor state from JSON file."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Clinical preprocessor config not found at: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
