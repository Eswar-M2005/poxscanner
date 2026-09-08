"""Dataset auditor for raw monkeypox / non-monkeypox lesion images.

Scan `data/raw/{monkeypox,non_monkeypox}`, validate with Pillow, detect
exact (SHA-256) and near (aHash, Hamming ≤ 5) duplicates, and write:

  ml/artifacts/metrics/audit_report.json
  ml/artifacts/metrics/audit_report.txt

CLI (from repo root)::

    python -m ml.src.audit
    python -m ml.src.audit --data-raw data/raw --output-dir ml/artifacts/metrics

Exit codes: 0 if at least one candidate image file was found, 2 if the
dataset is empty. Never fabricates images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

try:
    from .duplicates import (
        NEAR_DUPLICATE_HAMMING,
        average_hash_hex,
        cluster_by_hamming,
        hamming_distance,
    )
except ImportError:  # `python ml/src/audit.py` (not -m)
    from duplicates import (  # type: ignore[no-redef]
        NEAR_DUPLICATE_HAMMING,
        average_hash_hex,
        cluster_by_hamming,
        hamming_distance,
    )

try:
    from ml.src.utils import get_logger, repo_root, sha256_file
except ImportError:  # utils may be missing when this module is imported standalone

    def repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def get_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
        return logger


VERSION = "0.1.0"
CLASSES = ("non_monkeypox", "monkeypox")  # index 0, then 1
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
REPORT_JSON_NAME = "audit_report.json"
REPORT_TXT_NAME = "audit_report.txt"

EMPTY_STRUCTURE_MESSAGE = """\
No images found.

Expected directory structure (repo-relative):

  data/raw/monkeypox/         # class 1 images
  data/raw/non_monkeypox/     # class 0 images

Allowed types: .jpg .jpeg .png .webp .bmp

Add real images under those folders and re-run:
  python -m ml.src.audit

Do not synthesize training images.
"""

log = get_logger("ml.src.audit")


def _posix_relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _is_allowed_image(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_EXTENSIONS


def _iter_class_files(class_dir: Path) -> list[Path]:
    if not class_dir.is_dir():
        return []
    found: list[Path] = []
    for candidate in class_dir.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.name.startswith("."):
            continue
        if _is_allowed_image(candidate):
            found.append(candidate)
    found.sort()
    return found


def _inspect_image(path: Path, hasher: Callable[[Path], str]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": None,
        "bytes": path.stat().st_size,
        "sha256": hasher(path),
        "width": None,
        "height": None,
        "format": None,
        "mode": None,
        "ahash": None,
        "valid": False,
        "error": None,
    }
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            record["width"] = int(width)
            record["height"] = int(height)
            record["format"] = image.format
            record["mode"] = image.mode
            record["ahash"] = average_hash_hex(image)
            record["valid"] = True
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # pragma: no cover — unexpected decoder failures
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _imbalance_ratio(counts: dict[str, int]) -> float | None:
    positive = [value for value in counts.values() if value > 0]
    if len(positive) < 2:
        return None
    return max(positive) / min(positive)


def _cluster_max_hamming(ahashes: list[str]) -> int:
    ints = [int(value, 16) for value in ahashes]
    worst = 0
    for i, left in enumerate(ints):
        for right in ints[i + 1 :]:
            worst = max(worst, hamming_distance(left, right))
    return worst


def _duplicate_groups(
    files: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    valid_id_to_hash: dict[str, int] = {}
    by_path = {item["path"]: item for item in files}

    for item in files:
        by_sha[item["sha256"]].append(item)
        if item["valid"] and item["ahash"]:
            valid_id_to_hash[item["path"]] = int(item["ahash"], 16)

    exact: list[dict[str, Any]] = []
    cross_class_exact: list[dict[str, Any]] = []
    for digest, members in sorted(by_sha.items(), key=lambda kv: kv[0]):
        if len(members) < 2:
            continue
        labels = sorted({member["label"] for member in members})
        group = {
            "sha256": digest,
            "count": len(members),
            "paths": [member["path"] for member in members],
            "labels": labels,
            "cross_class": len(labels) > 1,
        }
        exact.append(group)
        if group["cross_class"]:
            cross_class_exact.append(group)

    near: list[dict[str, Any]] = []
    cross_class_near: list[dict[str, Any]] = []
    for cluster_paths in cluster_by_hamming(valid_id_to_hash, NEAR_DUPLICATE_HAMMING):
        members = [by_path[path] for path in cluster_paths]
        labels = sorted({member["label"] for member in members})
        ahashes = [member["ahash"] for member in members]
        shas = sorted({member["sha256"] for member in members})
        group = {
            "count": len(members),
            "paths": cluster_paths,
            "labels": labels,
            "ahashes": ahashes,
            "sha256s": shas,
            "max_hamming": _cluster_max_hamming(ahashes),
            "all_exact": len(shas) == 1,
            "cross_class": len(labels) > 1,
        }
        near.append(group)
        # Exact SHA cross-class is listed in cross_class_exact; here keep
        # perceptual copies that landed in both classes (re-encode / resize).
        if group["cross_class"] and not group["all_exact"]:
            cross_class_near.append(group)

    return exact, near, cross_class_exact, cross_class_near


def _empty_report(data_raw: Path, root: Path) -> dict[str, Any]:
    return {
        "status": "empty",
        "version": VERSION,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "data_raw": _posix_relative(data_raw, root),
        "classes": list(CLASSES),
        "allowed_extensions": list(ALLOWED_EXTENSIONS),
        "class_counts": {name: 0 for name in CLASSES},
        "class_counts_valid": {name: 0 for name in CLASSES},
        "class_counts_corrupt": {name: 0 for name in CLASSES},
        "total_files": 0,
        "valid_images": 0,
        "corrupt_count": 0,
        "imbalance_ratio": None,
        "format_counts": {},
        "files": [],
        "corrupt_files": [],
        "exact_duplicates": [],
        "near_duplicates": [],
        "cross_class_exact": [],
        "cross_class_near": [],
        "notes": [
            "No images found under data/raw/monkeypox and data/raw/non_monkeypox.",
            "Required layout is documented in docs/contracts.md. Do not fabricate images.",
        ],
    }


def _text_summary(report: dict[str, Any]) -> str:
    lines = [
        "Pox Scanner — dataset audit",
        "===========================",
        f"Status: {report['status']}",
        f"Version: {report['version']}",
        f"Scanned at (UTC): {report['scanned_at']}",
        f"Data root: {report['data_raw']}",
        "",
        "Class counts (candidate files):",
    ]
    for name in CLASSES:
        valid = report["class_counts_valid"][name]
        corrupt = report["class_counts_corrupt"][name]
        total = report["class_counts"][name]
        lines.append(f"  {name}: {total}  (valid={valid}, corrupt={corrupt})")
    ratio = report["imbalance_ratio"]
    ratio_text = "n/a (need both classes with valid images)" if ratio is None else f"{ratio:.4f} (majority / minority valid)"
    lines.extend(
        [
            "",
            f"Total files: {report['total_files']}",
            f"Valid images: {report['valid_images']}",
            f"Corrupt files: {report['corrupt_count']}",
            f"Imbalance ratio: {ratio_text}",
            f"Formats: {report['format_counts'] or '{}'}",
            "",
            f"Exact SHA-256 duplicate groups: {len(report['exact_duplicates'])}",
            f"Near-duplicate groups (aHash Hamming ≤ {NEAR_DUPLICATE_HAMMING}): {len(report['near_duplicates'])}",
            f"Cross-class identical SHA-256 groups: {len(report['cross_class_exact'])}",
            f"Cross-class near-duplicate groups: {len(report['cross_class_near'])}",
            "",
        ]
    )

    if report["status"] == "empty":
        lines.append(EMPTY_STRUCTURE_MESSAGE.rstrip())
        lines.append("")
        return "\n".join(lines) + "\n"

    if report["corrupt_files"]:
        lines.append("Corrupt files:")
        for item in report["corrupt_files"]:
            lines.append(f"  - {item['path']}: {item['error']}")
        lines.append("")

    if report["cross_class_exact"]:
        lines.append("WARNING: identical files appear in both classes (same SHA-256):")
        for group in report["cross_class_exact"]:
            lines.append(f"  sha256={group['sha256']}")
            for path in group["paths"]:
                lines.append(f"    {path}")
        lines.append("")

    if report["cross_class_near"]:
        lines.append("WARNING: near-duplicate images appear in both classes (aHash):")
        for group in report["cross_class_near"]:
            lines.append(f"  max_hamming={group['max_hamming']} paths={group['paths']}")
        lines.append("")

    if report["exact_duplicates"]:
        lines.append("Exact duplicates:")
        for group in report["exact_duplicates"]:
            cross = " [CROSS-CLASS]" if group["cross_class"] else ""
            lines.append(f"  {group['sha256'][:12]}… n={group['count']}{cross}")
            for path in group["paths"]:
                lines.append(f"    {path}")
        lines.append("")

    if report["near_duplicates"]:
        lines.append("Near-duplicates:")
        for group in report["near_duplicates"]:
            kind = "exact-sha" if group["all_exact"] else "perceptual"
            cross = " [CROSS-CLASS]" if group["cross_class"] else ""
            lines.append(
                f"  {kind} n={group['count']} max_hamming={group['max_hamming']}{cross}"
            )
            for path in group["paths"]:
                lines.append(f"    {path}")
        lines.append("")

    for note in report["notes"]:
        lines.append(f"Note: {note}")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_reports(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / REPORT_JSON_NAME
    txt_path = output_dir / REPORT_TXT_NAME
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    txt_path.write_text(_text_summary(report), encoding="utf-8")
    return json_path, txt_path


def run_audit(
    *,
    data_raw: Path | None = None,
    output_dir: Path | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Scan class folders, write reports, and return the JSON-serializable dict."""
    repo = Path(root) if root is not None else Path(repo_root())
    raw = Path(data_raw) if data_raw is not None else repo / "data" / "raw"
    metrics = (
        Path(output_dir)
        if output_dir is not None
        else repo / "ml" / "artifacts" / "metrics"
    )

    files: list[dict[str, Any]] = []
    class_counts = {name: 0 for name in CLASSES}
    class_counts_valid = {name: 0 for name in CLASSES}
    class_counts_corrupt = {name: 0 for name in CLASSES}

    for label in CLASSES:
        class_dir = raw / label
        for path in _iter_class_files(class_dir):
            item = _inspect_image(path, sha256_file)
            item["path"] = _posix_relative(path, repo)
            item["label"] = label
            files.append(item)
            class_counts[label] += 1
            if item["valid"]:
                class_counts_valid[label] += 1
            else:
                class_counts_corrupt[label] += 1

    if not files:
        report = _empty_report(raw, repo)
        json_path, txt_path = write_reports(report, metrics)
        report["_json_path"] = json_path.as_posix()
        report["_txt_path"] = txt_path.as_posix()
        return report

    exact, near, cross_exact, cross_near = _duplicate_groups(files)

    format_counts: dict[str, int] = defaultdict(int)
    for item in files:
        if item["valid"] and item["format"]:
            format_counts[item["format"]] += 1

    notes: list[str] = []
    missing_dirs = [name for name in CLASSES if not (raw / name).is_dir()]
    if missing_dirs:
        notes.append("Missing class directories: " + ", ".join(missing_dirs))
    if any(class_counts_valid[name] == 0 for name in CLASSES):
        empty_classes = [name for name in CLASSES if class_counts_valid[name] == 0]
        notes.append("No valid images for: " + ", ".join(empty_classes))
    if cross_exact:
        notes.append(
            "Suspiciously identical files (same SHA-256) appear in both classes."
        )
    if cross_near:
        notes.append(
            "Near-duplicate images (aHash Hamming ≤ "
            f"{NEAR_DUPLICATE_HAMMING}) appear in both classes."
        )

    public_files = [
        {
            "path": item["path"],
            "label": item["label"],
            "width": item["width"],
            "height": item["height"],
            "format": item["format"],
            "mode": item["mode"],
            "bytes": item["bytes"],
            "sha256": item["sha256"],
            "ahash": item["ahash"],
            "valid": item["valid"],
        }
        for item in files
    ]
    corrupt_files = [
        {
            "path": item["path"],
            "label": item["label"],
            "bytes": item["bytes"],
            "sha256": item["sha256"],
            "error": item["error"],
        }
        for item in files
        if not item["valid"]
    ]

    has_issues = bool(corrupt_files or cross_exact or cross_near)
    report: dict[str, Any] = {
        "status": "issues" if has_issues else "ok",
        "version": VERSION,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "data_raw": _posix_relative(raw, repo),
        "classes": list(CLASSES),
        "allowed_extensions": list(ALLOWED_EXTENSIONS),
        "class_counts": class_counts,
        "class_counts_valid": class_counts_valid,
        "class_counts_corrupt": class_counts_corrupt,
        "total_files": len(files),
        "valid_images": sum(class_counts_valid.values()),
        "corrupt_count": len(corrupt_files),
        "imbalance_ratio": _imbalance_ratio(class_counts_valid),
        "format_counts": dict(sorted(format_counts.items())),
        "files": public_files,
        "corrupt_files": corrupt_files,
        "exact_duplicates": exact,
        "near_duplicates": near,
        "cross_class_exact": cross_exact,
        "cross_class_near": cross_near,
        "notes": notes,
    }
    json_path, txt_path = write_reports(report, metrics)
    report["_json_path"] = json_path.as_posix()
    report["_txt_path"] = txt_path.as_posix()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the raw lesion-image dataset (no image fabrication)."
    )
    parser.add_argument(
        "--data-raw",
        type=Path,
        default=None,
        help="Class-folder parent (default: <repo>/data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write audit_report.json/txt (default: ml/artifacts/metrics)",
    )
    args = parser.parse_args(argv)

    report = run_audit(data_raw=args.data_raw, output_dir=args.output_dir)
    json_path = report.get("_json_path")
    txt_path = report.get("_txt_path")

    if report["status"] == "empty":
        print(EMPTY_STRUCTURE_MESSAGE, file=sys.stderr)
        if json_path:
            print(f"Wrote empty audit report: {json_path}", file=sys.stderr)
        log.error("Dataset empty — refusing to invent images.")
        return 2

    print(_text_summary(report), end="")
    if json_path:
        print(f"JSON: {json_path}")
    if txt_path:
        print(f"Text: {txt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
