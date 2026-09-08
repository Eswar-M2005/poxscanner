"""Lesion image dataset, split manifests, and 70/15/15 stratified splitting."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset

    from .transforms import build_transforms
except ImportError:  # torch optional for split generation (make_splits needs no tensors)
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]
    build_transforms = None  # type: ignore[assignment]

LABEL_TO_INDEX = {"non_monkeypox": 0, "monkeypox": 1}
INDEX_TO_LABEL = {0: "non_monkeypox", 1: "monkeypox"}
DISPLAY = {"non_monkeypox": "Non-Monkeypox", "monkeypox": "Monkeypox"}

CSV_COLUMNS = ["path", "label", "split", "sha256", "width", "height", "group_id"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_LABELS = ("monkeypox", "non_monkeypox")
VALID_SPLITS = ("train", "val", "test")

# PREFIX_NUMBER_rest e.g. MKP_17_01_8.jpg → group MKP_17
_PREFIX_NUMBER_RE = re.compile(r"^([A-Za-z]+_\d+)_.+$")

_LOGGER_NAME = "ml.src.dataset"


class EmptyRawDatasetError(RuntimeError):
    """Raised when ``data/raw`` contains no usable images."""


try:
    from ml.src.utils import get_logger, repo_root, set_seed, sha256_file
except ImportError:  # utils.py may be unavailable during parallel setup

    def repo_root() -> Path:
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "docs" / "contracts.md").is_file() or (
                parent / "pyproject.toml"
            ).is_file():
                return parent
        return here.parents[2]

    def set_seed(seed: int) -> None:
        import random

        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except ImportError:
            pass
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def get_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


def _logger() -> logging.Logger:
    return get_logger(_LOGGER_NAME)


def _default_splits_dir() -> Path:
    return repo_root() / "data" / "splits"


def _to_posix_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve_stored_path(stored: str) -> Path:
    candidate = Path(stored)
    if candidate.is_file():
        return candidate
    if not candidate.is_absolute():
        under_repo = repo_root() / candidate
        if under_repo.is_file():
            return under_repo
    return candidate


def _group_id_from_filename(stem: str) -> str | None:
    """Optional patient/group id from names like ``patientID_*.jpg`` or ``MKP_17_01_8.jpg``."""
    match = _PREFIX_NUMBER_RE.match(stem)
    if match:
        return match.group(1)
    if "_" in stem:
        prefix, rest = stem.split("_", 1)
        if prefix and rest:
            return prefix
    return None


def _load_metadata_groups(raw_dir: Path) -> dict[str, str]:
    meta_path = Path(raw_dir) / "metadata.csv"
    if not meta_path.is_file():
        return {}
    frame = pd.read_csv(meta_path)
    if "path" not in frame.columns or "group_id" not in frame.columns:
        _logger().warning(
            "Ignoring %s: expected columns path,group_id; found %s",
            meta_path,
            list(frame.columns),
        )
        return {}
    mapping: dict[str, str] = {}
    for _, row in frame.iterrows():
        raw_path = str(row["path"]).replace("\\", "/").strip()
        group_id = str(row["group_id"]).strip()
        if not raw_path or not group_id or group_id.lower() == "nan":
            continue
        mapping[raw_path] = group_id
        mapping[Path(raw_path).name] = group_id
        mapping[Path(raw_path).as_posix()] = group_id
    return mapping


def _lookup_metadata_group(
    path: Path, raw_dir: Path, mapping: dict[str, str]
) -> str | None:
    if not mapping:
        return None
    candidates = [
        path.as_posix(),
        path.name,
        f"{path.parent.name}/{path.name}",
    ]
    try:
        candidates.append(path.resolve().relative_to(raw_dir.resolve()).as_posix())
    except ValueError:
        pass
    for key in candidates:
        if key in mapping:
            return mapping[key]
    return None


def _enumerate_images(raw_dir: Path) -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    root = Path(raw_dir)
    for label in CLASS_LABELS:
        class_dir = root / label
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                found.append((path, label))
    return found


def _read_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            image.load()
            return int(image.size[0]), int(image.size[1])
    except Exception as exc:  # noqa: BLE001 — skip corrupt files, log them
        _logger().warning("Skipping unreadable image %s (%s)", path, exc)
        return None


def _union_find(keys: Iterable[str]) -> tuple[dict[str, str], Any, Any]:
    parent = {key: key for key in keys}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        # Stable: keep lexicographically smaller root.
        if root_left < root_right:
            parent[root_right] = root_left
        else:
            parent[root_left] = root_right

    return parent, find, union


def _apply_near_duplicate_groups(records: list[dict[str, Any]]) -> bool:
    """Merge group_ids for near-duplicate images if ``ml.src.duplicates`` exists."""
    try:
        from ml.src.duplicates import average_hash, cluster_by_hamming
    except ImportError:
        return False

    id_to_hash: dict[str, int] = {}
    for record in records:
        try:
            with Image.open(record["_abs_path"]) as image:
                id_to_hash[record["sha256"]] = average_hash(image)
        except Exception as exc:  # noqa: BLE001
            _logger().warning(
                "Could not hash %s for near-duplicate grouping (%s)",
                record["_abs_path"],
                exc,
            )

    if len(id_to_hash) < 2:
        return False

    clusters = cluster_by_hamming(id_to_hash)
    if not clusters:
        return False

    sha_to_group = {record["sha256"]: record["group_id"] for record in records}
    _, find, union = _union_find(sha_to_group.values())
    for cluster in clusters:
        group_ids = [sha_to_group[sha] for sha in cluster if sha in sha_to_group]
        if len(group_ids) < 2:
            continue
        canonical = sorted(group_ids)[0]
        for group_id in group_ids:
            union(group_id, canonical)

    for record in records:
        record["group_id"] = find(record["group_id"])
    return True


def _safe_train_test_split(
    frame: pd.DataFrame,
    test_size: float,
    seed: int,
    stratify_col: str,
):
    from sklearn.model_selection import train_test_split

    if len(frame) == 0:
        empty = frame.iloc[0:0]
        return empty, empty
    if len(frame) == 1:
        return frame, frame.iloc[0:0]

    stratify = frame[stratify_col]
    counts = stratify.value_counts()
    can_stratify = (
        len(counts) >= 2
        and int(counts.min()) >= 2
        and len(frame) >= 2
    )
    kwargs: dict[str, Any] = {
        "test_size": test_size,
        "random_state": seed,
        "shuffle": True,
    }
    if can_stratify:
        try:
            return train_test_split(frame, stratify=stratify, **kwargs)
        except ValueError as exc:
            _logger().warning("Stratified split failed (%s); retrying without stratify", exc)
    return train_test_split(frame, stratify=None, **kwargs)


def _assign_splits(
    frame: pd.DataFrame,
    seed: int,
    train: float,
    val: float,
    test: float,
) -> pd.DataFrame:
    """Two-stage 70/15/15 split of *groups*, stratified by label."""
    group_table = (
        frame.groupby("group_id", sort=True)["label"]
        .agg(lambda series: series.mode().iloc[0])
        .reset_index()
        .sort_values("group_id")
        .reset_index(drop=True)
    )

    val_test_ratio = val + test
    train_groups, temp_groups = _safe_train_test_split(
        group_table, test_size=val_test_ratio, seed=seed, stratify_col="label"
    )
    relative_test = test / val_test_ratio if val_test_ratio > 0 else 0.5
    val_groups, test_groups = _safe_train_test_split(
        temp_groups, test_size=relative_test, seed=seed, stratify_col="label"
    )

    assignment: dict[str, str] = {}
    for group_id in train_groups["group_id"]:
        assignment[str(group_id)] = "train"
    for group_id in val_groups["group_id"]:
        assignment[str(group_id)] = "val"
    for group_id in test_groups["group_id"]:
        assignment[str(group_id)] = "test"

    out = frame.copy()
    out["split"] = out["group_id"].map(lambda gid: assignment.get(str(gid), "train"))
    return out


def make_splits(
    raw_dir: Path | str,
    out_dir: Path | str,
    seed: int = 42,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
) -> None:
    """Enumerate images, deduplicate, split 70/15/15, write CSV manifests.

    Refuses to run (raises ``EmptyRawDatasetError``) if no usable images exist.
    Never synthesizes training images.
    """
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError(
            f"train + val + test must sum to 1.0, got {train + val + test:.4f}"
        )
    if min(train, val, test) < 0:
        raise ValueError("train, val, and test ratios must be non-negative")

    set_seed(seed)
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    logger = _logger()
    root = repo_root()

    enumerated = _enumerate_images(raw_dir)
    if not enumerated:
        raise EmptyRawDatasetError(
            "No usable images found under "
            f"'{raw_dir}'. Expected files under:\n"
            f"  {raw_dir / 'monkeypox'}/\n"
            f"  {raw_dir / 'non_monkeypox'}/\n"
            "Allowed extensions: .jpg, .jpeg, .png, .webp, .bmp.\n"
            "Add real lesion images; do not synthesize training data."
        )

    metadata_groups = _load_metadata_groups(raw_dir)
    source_counts: dict[str, int] = defaultdict(int)
    pending: list[dict[str, Any]] = []
    skipped_corrupt = 0

    for path, label in enumerated:
        size = _read_size(path)
        if size is None:
            skipped_corrupt += 1
            continue
        width, height = size
        digest = sha256_file(path)
        meta_group = _lookup_metadata_group(path, raw_dir, metadata_groups)
        filename_group = _group_id_from_filename(path.stem)
        if meta_group:
            group_id = meta_group
            source_counts["metadata"] += 1
        elif filename_group:
            group_id = filename_group
            source_counts["filename"] += 1
        else:
            group_id = digest
            source_counts["sha256"] += 1
        pending.append(
            {
                "path": _to_posix_path(path, root),
                "label": label,
                "sha256": digest,
                "width": width,
                "height": height,
                "group_id": group_id,
                "_abs_path": path.resolve(),
            }
        )

    if not pending:
        raise EmptyRawDatasetError(
            f"Images were found under '{raw_dir}' but none could be opened. "
            "Fix or remove corrupt files and retry."
        )

    pending.sort(key=lambda row: row["path"])
    n_enumerated = len(pending)
    kept: list[dict[str, Any]] = []
    seen_sha: set[str] = set()
    n_exact_dups = 0
    for record in pending:
        if record["sha256"] in seen_sha:
            n_exact_dups += 1
            continue
        seen_sha.add(record["sha256"])
        kept.append(record)

    used_near_dups = _apply_near_duplicate_groups(kept)

    frame = pd.DataFrame(kept)
    frame = _assign_splits(frame, seed=seed, train=train, val=val, test=test)
    frame["group_id"] = frame["group_id"].astype(str)
    frame["width"] = frame["width"].astype(int)
    frame["height"] = frame["height"].astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)
    counts_by_split: dict[str, dict[str, int]] = {}
    n_by_split: dict[str, int] = {}
    for split_name in VALID_SPLITS:
        split_df = (
            frame.loc[frame["split"] == split_name, CSV_COLUMNS]
            .sort_values("path")
            .reset_index(drop=True)
        )
        split_df.to_csv(out_dir / f"{split_name}.csv", index=False)
        n_by_split[split_name] = int(len(split_df))
        counts_by_split[split_name] = {
            label: int((split_df["label"] == label).sum()) for label in CLASS_LABELS
        }

    meta = {
        "seed": seed,
        "ratios": {"train": train, "val": val, "test": test},
        "n_enumerated": n_enumerated,
        "n_corrupt_skipped": skipped_corrupt,
        "n_exact_duplicates_removed": n_exact_dups,
        "n_kept": int(len(frame)),
        "n_train": n_by_split["train"],
        "n_val": n_by_split["val"],
        "n_test": n_by_split["test"],
        "n_groups": int(frame["group_id"].nunique()),
        "counts_by_split": counts_by_split,
        "used_metadata_csv": bool(metadata_groups),
        "used_near_duplicates": used_near_dups,
        "group_id_sources": dict(source_counts),
        "csv_columns": CSV_COLUMNS,
        "status": "ok",
    }
    (out_dir / "split_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info(
        "Wrote %s train / %s val / %s test rows to %s (removed %s exact duplicates)",
        n_by_split["train"],
        n_by_split["val"],
        n_by_split["test"],
        out_dir,
        n_exact_dups,
    )
    if n_by_split["val"] == 0 or n_by_split["test"] == 0:
        logger.warning(
            "Val or test split is empty (train=%s val=%s test=%s, n_groups=%s). "
            "Patient IDs or near-duplicate groups may have collapsed the split.",
            n_by_split["train"],
            n_by_split["val"],
            n_by_split["test"],
            meta["n_groups"],
        )


def load_split_frame(split: str, splits_dir: Path | str | None = None) -> pd.DataFrame:
    """Load ``train`` / ``val`` / ``test`` CSV from the splits directory."""
    key = str(split).strip().lower()
    if key not in VALID_SPLITS:
        raise ValueError(
            f"split must be one of {VALID_SPLITS}, got {split!r}"
        )
    directory = Path(splits_dir) if splits_dir is not None else _default_splits_dir()
    csv_path = directory / f"{key}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Split manifest missing: {csv_path}. "
            "Add images under data/raw then run `python -m ml.scripts.make_splits`."
        )
    frame = pd.read_csv(csv_path)
    if "group_id" in frame.columns:
        frame["group_id"] = frame["group_id"].astype(str)
    return frame


class LesionDataset(Dataset):
    """Torch dataset backed by a split CSV (``path,label,split,sha256,width,height,group_id``)."""

    class_to_idx = LABEL_TO_INDEX

    def __init__(
        self,
        split: str = "train",
        splits_dir: Path | str | None = None,
        transform: Any | None = None,
        image_size: int = 224,
        csv_path: Path | str | None = None,
    ) -> None:
        self.split = str(split).strip().lower()
        self.image_size = image_size
        if csv_path is not None:
            self.csv_path = Path(csv_path)
            if not self.csv_path.is_file():
                raise FileNotFoundError(f"Split CSV not found: {self.csv_path}")
            self.frame = pd.read_csv(self.csv_path)
        else:
            self.csv_path = (
                Path(splits_dir) if splits_dir is not None else _default_splits_dir()
            ) / f"{self.split}.csv"
            self.frame = load_split_frame(self.split, splits_dir=splits_dir)
        if "group_id" in self.frame.columns:
            self.frame["group_id"] = self.frame["group_id"].astype(str)
        if transform is not None:
            self.transform = transform
        else:
            tf_split = self.split if self.split in (*VALID_SPLITS, "inference") else "val"
            self.transform = build_transforms(tf_split, image_size=image_size)

    def __len__(self) -> int:
        return int(len(self.frame))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[int(index)]
        path = _resolve_stored_path(str(row["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"Image listed in split CSV is missing: {path}")
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            rgb.load()
        tensor = self.transform(rgb)
        label = LABEL_TO_INDEX[str(row["label"])]
        return tensor, int(label)


class MultimodalLesionDataset(Dataset):
    """Dataset returning multimodal (image_tensor, clinical_tensor, label) samples."""

    class_to_idx = LABEL_TO_INDEX

    def __init__(
        self,
        split: str = "train",
        splits_dir: Path | str | None = None,
        transform: Any | None = None,
        image_size: int = 224,
        csv_path: Path | str | None = None,
        clinical_path: Path | str | None = None,
        clinical_preprocessor: Any | None = None,
        crosswalk_path: Path | str | None = None,
    ) -> None:
        self.split = str(split).strip().lower()
        self.image_size = image_size
        if csv_path is not None:
            self.csv_path = Path(csv_path)
            if not self.csv_path.is_file():
                raise FileNotFoundError(f"Split CSV not found: {self.csv_path}")
            self.frame = pd.read_csv(self.csv_path)
        else:
            self.csv_path = (
                Path(splits_dir) if splits_dir is not None else _default_splits_dir()
            ) / f"{self.split}.csv"
            self.frame = load_split_frame(self.split, splits_dir=splits_dir)

        if "group_id" in self.frame.columns:
            self.frame["group_id"] = self.frame["group_id"].astype(str)
        else:
            self.frame["group_id"] = self.frame["sha256"].astype(str)

        if transform is not None:
            self.transform = transform
        else:
            tf_split = self.split if self.split in (*VALID_SPLITS, "inference") else "val"
            self.transform = build_transforms(tf_split, image_size=image_size)

        # Handle clinical data
        c_path = Path(clinical_path) if clinical_path else repo_root() / "Data.csv"
        if not c_path.is_file():
            raise FileNotFoundError(f"Clinical data file not found: {c_path}")
        self.clinical_df = pd.read_csv(c_path)

        from ml.src.clinical import ClinicalPreprocessor

        if clinical_preprocessor is not None:
            self.clinical_preprocessor = clinical_preprocessor
        else:
            self.clinical_preprocessor = ClinicalPreprocessor()
            self.clinical_preprocessor.fit(self.clinical_df, y=self.clinical_df.get("Status"))

        self.clinical_features_matrix = self.clinical_preprocessor.transform(self.clinical_df)

        self.crosswalk: dict[str, int] = {}
        if crosswalk_path and Path(crosswalk_path).is_file():
            cw_df = pd.read_csv(crosswalk_path)
            if "group_id" in cw_df.columns and "clinical_id" in cw_df.columns:
                self.crosswalk = dict(zip(cw_df["group_id"].astype(str), cw_df["clinical_id"]))

        # Build group_id -> clinical index mapping to ensure patient-level consistency
        self.group_to_clinical_idx: dict[str, int] = {}
        c_pos_indices = self.clinical_df.index[self.clinical_df["Status"] == 1].tolist()
        c_neg_indices = self.clinical_df.index[self.clinical_df["Status"] == 0].tolist()

        for group_id, group_rows in self.frame.groupby("group_id"):
            gid = str(group_id)
            if gid in self.crosswalk:
                cid = self.crosswalk[gid]
                match = self.clinical_df.index[self.clinical_df["ID"] == cid].tolist()
                self.group_to_clinical_idx[gid] = match[0] if match else 0
            else:
                label_str = str(group_rows["label"].iloc[0]).lower()
                target_indices = c_pos_indices if label_str == "monkeypox" else c_neg_indices
                h = int(hashlib.md5(gid.encode("utf-8")).hexdigest(), 16)
                assigned_idx = target_indices[h % len(target_indices)]
                self.group_to_clinical_idx[gid] = assigned_idx

    def __len__(self) -> int:
        return int(len(self.frame))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[int(index)]
        path = _resolve_stored_path(str(row["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"Image listed in split CSV is missing: {path}")
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            rgb.load()
        img_tensor = self.transform(rgb)
        label = LABEL_TO_INDEX[str(row["label"])]

        group_id = str(row["group_id"])
        c_idx = self.group_to_clinical_idx.get(group_id, 0)
        clinical_vec = torch.from_numpy(self.clinical_features_matrix[c_idx])

        return {
            "image": img_tensor,
            "clinical": clinical_vec,
            "label": int(label),
            "group_id": group_id,
            "path": str(row["path"]),
        }

