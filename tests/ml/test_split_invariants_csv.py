"""CSV-only split invariants for data/splits/*.csv — no image IO, no torch.

Skips when train/val/test manifests are missing (fresh checkout without data).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[2]
_SPLITS_DIR = _REPO / "data" / "splits"
_SPLIT_NAMES = ("train", "val", "test")
_CSV_PATHS = {name: _SPLITS_DIR / f"{name}.csv" for name in _SPLIT_NAMES}

REQUIRED_COLUMNS = ["path", "label", "split", "sha256", "width", "height", "group_id"]
EXPECTED_COUNTS = {"train": 6374, "val": 272, "test": 270}
EXPECTED_TOTAL = 6916  # 6374 + 272 + 270


def _csvs_present() -> bool:
    return all(path.is_file() for path in _CSV_PATHS.values())


pytestmark = pytest.mark.skipif(
    not _csvs_present(),
    reason="data/splits/{train,val,test}.csv not present",
)


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    loaded: dict[str, pd.DataFrame] = {}
    for name, path in _CSV_PATHS.items():
        loaded[name] = pd.read_csv(path)
    return loaded


def test_row_counts_match_frozen_totals(frames: dict[str, pd.DataFrame]) -> None:
    counts = {name: len(frame) for name, frame in frames.items()}
    assert counts == EXPECTED_COUNTS
    assert sum(counts.values()) == EXPECTED_TOTAL
    assert EXPECTED_COUNTS["train"] + EXPECTED_COUNTS["val"] + EXPECTED_COUNTS["test"] == EXPECTED_TOTAL


def test_required_columns_present(frames: dict[str, pd.DataFrame]) -> None:
    for name, frame in frames.items():
        missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
        assert not missing, f"{name}.csv missing columns: {missing}"


def test_unique_sha256_across_splits(frames: dict[str, pd.DataFrame]) -> None:
    by_split = {
        name: set(frame["sha256"].dropna().astype(str)) for name, frame in frames.items()
    }
    assert by_split["train"].isdisjoint(by_split["val"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["val"].isdisjoint(by_split["test"])
    combined = pd.concat(list(frames.values()), ignore_index=True)
    assert combined["sha256"].isna().sum() == 0
    assert combined["sha256"].duplicated().sum() == 0
    assert len(by_split["train"] | by_split["val"] | by_split["test"]) == EXPECTED_TOTAL


def test_unique_group_id_across_splits(frames: dict[str, pd.DataFrame]) -> None:
    by_split = {
        name: set(frame["group_id"].dropna().astype(str)) for name, frame in frames.items()
    }
    assert by_split["train"].isdisjoint(by_split["val"]), "group_id leaked train↔val"
    assert by_split["train"].isdisjoint(by_split["test"]), "group_id leaked train↔test"
    assert by_split["val"].isdisjoint(by_split["test"]), "group_id leaked val↔test"
    combined = pd.concat(list(frames.values()), ignore_index=True)
    assert combined["group_id"].isna().sum() == 0
    for group_id, group in combined.groupby("group_id"):
        splits = set(group["split"])
        assert len(splits) == 1, f"group_id {group_id!r} appears in splits {splits}"
