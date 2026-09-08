"""Split generation: 70/15/15-ish, no sha256 leakage, stratified by label."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml.src.dataset import (  # noqa: E402
    CSV_COLUMNS,
    DISPLAY,
    INDEX_TO_LABEL,
    LABEL_TO_INDEX,
    EmptyRawDatasetError,
    LesionDataset,
    load_split_frame,
    make_splits,
)
from ml.src.transforms import build_transforms  # noqa: E402
from ml.scripts.make_splits import main as make_splits_main  # noqa: E402

N_PER_CLASS = 10


def _write_png(path: Path, code_index: int) -> None:
    """Write a 32×32 PNG whose 8×8 aHash is 3 unique bits (pairwise Hamming = 6).

    Near-duplicate clustering uses Hamming ≤ 5 and is transitive, so test
    fixtures must not look alike or every image collapses into one split.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (32, 32), (10, 10, 10))
    pixels = image.load()
    for bit in (3 * code_index, 3 * code_index + 1, 3 * code_index + 2):
        block_y, block_x = divmod(bit, 8)
        for y in range(block_y * 4, block_y * 4 + 4):
            for x in range(block_x * 4, block_x * 4 + 4):
                pixels[x, y] = (240, 240, 240)
    image.save(path)


def _tiny_raw(tmp_path: Path, n_per_class: int = N_PER_CLASS) -> Path:
    raw = tmp_path / "raw"
    for i in range(n_per_class):
        _write_png(raw / "monkeypox" / f"mpox{i:02d}.png", code_index=i)
        _write_png(
            raw / "non_monkeypox" / f"nmpox{i:02d}.png",
            code_index=n_per_class + i,
        )
    return raw


def _read_all(out: Path) -> pd.DataFrame:
    frames = [pd.read_csv(out / f"{name}.csv") for name in ("train", "val", "test")]
    return pd.concat(frames, ignore_index=True)


def test_label_maps_match_contracts() -> None:
    assert LABEL_TO_INDEX == {"non_monkeypox": 0, "monkeypox": 1}
    assert INDEX_TO_LABEL == {0: "non_monkeypox", 1: "monkeypox"}
    assert DISPLAY == {"non_monkeypox": "Non-Monkeypox", "monkeypox": "Monkeypox"}


def test_make_splits_ratios_no_sha_overlap(tmp_path: Path) -> None:
    raw = _tiny_raw(tmp_path)
    out = tmp_path / "splits"
    make_splits(raw, out, seed=42)

    train = pd.read_csv(out / "train.csv")
    val = pd.read_csv(out / "val.csv")
    test = pd.read_csv(out / "test.csv")
    total = len(train) + len(val) + len(test)
    assert total == N_PER_CLASS * 2
    assert abs(len(train) / total - 0.70) <= 0.15
    assert abs(len(val) / total - 0.15) <= 0.15
    assert abs(len(test) / total - 0.15) <= 0.15
    # Exact 14/3/3 on 20 unique files with two-stage 0.30 then 0.50
    assert (len(train), len(val), len(test)) == (14, 3, 3)

    hashes = {
        "train": set(train["sha256"]),
        "val": set(val["sha256"]),
        "test": set(test["sha256"]),
    }
    assert hashes["train"].isdisjoint(hashes["val"])
    assert hashes["train"].isdisjoint(hashes["test"])
    assert hashes["val"].isdisjoint(hashes["test"])
    assert len(hashes["train"] | hashes["val"] | hashes["test"]) == total

    for frame in (train, val, test):
        assert list(frame.columns) == CSV_COLUMNS

    meta = json.loads((out / "split_meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 42
    assert meta["n_kept"] == total
    assert meta["n_exact_duplicates_removed"] == 0


def test_make_splits_stratified(tmp_path: Path) -> None:
    raw = _tiny_raw(tmp_path)
    out = tmp_path / "splits"
    make_splits(raw, out, seed=42)
    for name in ("train", "val", "test"):
        frame = pd.read_csv(out / f"{name}.csv")
        labels = set(frame["label"])
        assert labels == {"monkeypox", "non_monkeypox"}, f"{name} missing a class: {labels}"
        assert frame["split"].nunique() == 1
        assert frame["split"].iloc[0] == name


def test_make_splits_empty_raises(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "monkeypox").mkdir(parents=True)
    (raw / "non_monkeypox").mkdir(parents=True)
    with pytest.raises(EmptyRawDatasetError, match="No usable images"):
        make_splits(raw, tmp_path / "splits")


def test_cli_empty_exits_nonzero(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "monkeypox").mkdir(parents=True)
    (raw / "non_monkeypox").mkdir(parents=True)
    code = make_splits_main(
        ["--raw-dir", str(raw), "--out-dir", str(tmp_path / "splits")]
    )
    assert code == 1


def test_exact_duplicates_keep_one(tmp_path: Path) -> None:
    raw = _tiny_raw(tmp_path, n_per_class=10)
    # Identical bytes → same sha256
    _write_png(raw / "monkeypox" / "mpox00.png", code_index=0)
    _write_png(raw / "monkeypox" / "mpox00_copy.png", code_index=0)
    out = tmp_path / "splits"
    make_splits(raw, out, seed=42)
    combined = _read_all(out)
    assert combined["sha256"].duplicated().sum() == 0
    meta = json.loads((out / "split_meta.json").read_text(encoding="utf-8"))
    assert meta["n_exact_duplicates_removed"] >= 1


def test_patient_filename_group_stays_together(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    # Two images from patient p99 plus enough unique files for a split
    _write_png(raw / "monkeypox" / "p99_a.png", code_index=16)
    _write_png(raw / "monkeypox" / "p99_b.png", code_index=17)
    for i in range(8):
        _write_png(raw / "monkeypox" / f"mpox{i:02d}.png", code_index=i)
        _write_png(raw / "non_monkeypox" / f"nmpox{i:02d}.png", code_index=8 + i)
    out = tmp_path / "splits"
    make_splits(raw, out, seed=42)
    combined = _read_all(out)
    pair = combined[combined["path"].str.contains("p99_")]
    assert len(pair) == 2
    assert pair["group_id"].nunique() == 1
    assert pair["split"].nunique() == 1


def test_metadata_csv_group_id(tmp_path: Path) -> None:
    raw = _tiny_raw(tmp_path, n_per_class=8)
    rows = [
        {"path": "monkeypox/mpox00.png", "group_id": "patientA"},
        {"path": "monkeypox/mpox01.png", "group_id": "patientA"},
    ]
    pd.DataFrame(rows).to_csv(raw / "metadata.csv", index=False)
    out = tmp_path / "splits"
    make_splits(raw, out, seed=42)
    combined = _read_all(out)
    grouped = combined[combined["group_id"] == "patientA"]
    assert len(grouped) == 2
    assert grouped["split"].nunique() == 1
    meta = json.loads((out / "split_meta.json").read_text(encoding="utf-8"))
    assert meta["used_metadata_csv"] is True


def test_load_split_frame_and_dataset(tmp_path: Path) -> None:
    raw = _tiny_raw(tmp_path)
    out = tmp_path / "splits"
    make_splits(raw, out, seed=42)
    frame = load_split_frame("train", splits_dir=out)
    assert len(frame) == 14
    assert set(frame["label"]) <= {"monkeypox", "non_monkeypox"}

    dataset = LesionDataset(
        split="val",
        splits_dir=out,
        transform=build_transforms("val", image_size=32),
        image_size=32,
    )
    assert len(dataset) > 0
    image, label = dataset[0]
    assert tuple(image.shape) == (3, 32, 32)
    assert label in (0, 1)


def test_load_split_frame_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Split manifest missing"):
        load_split_frame("train", splits_dir=tmp_path / "nope")
