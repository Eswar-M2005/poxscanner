"""Tests for the dataset auditor. Uses tiny generated PNGs only — no real dataset."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.src.audit import (  # noqa: E402
    EMPTY_STRUCTURE_MESSAGE,
    main,
    run_audit,
)
from ml.src.duplicates import (  # noqa: E402
    NEAR_DUPLICATE_HAMMING,
    average_hash,
    average_hash_hex,
    cluster_by_hamming,
    hamming_distance,
)


def _write_pattern_png(path: Path, kind: str, size: tuple[int, int] = (32, 32)) -> None:
    """Structured patterns so aHash differs (solid colors all hash to 0)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, (0, 0, 0))
    draw = ImageDraw.Draw(image)
    w, h = size
    if kind == "left":
        draw.rectangle([0, 0, w // 2, h], fill=(255, 255, 255))
    elif kind == "top":
        draw.rectangle([0, 0, w, h // 2], fill=(255, 255, 255))
    else:
        raise ValueError(kind)
    image.save(path, format="PNG")


def _layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw = tmp_path / "data" / "raw"
    (raw / "monkeypox").mkdir(parents=True)
    (raw / "non_monkeypox").mkdir(parents=True)
    metrics = tmp_path / "ml" / "artifacts" / "metrics"
    return raw, metrics, tmp_path


def test_audit_counts_valid_and_corrupt(tmp_path: Path) -> None:
    raw, metrics, root = _layout(tmp_path)
    _write_pattern_png(raw / "monkeypox" / "a.png", "left")
    _write_pattern_png(raw / "non_monkeypox" / "b.png", "top")
    corrupt = raw / "monkeypox" / "broken.png"
    corrupt.write_bytes(b"this is not a png")

    report = run_audit(data_raw=raw, output_dir=metrics, root=root)

    assert report["status"] == "issues"
    assert report["class_counts"]["monkeypox"] == 2
    assert report["class_counts"]["non_monkeypox"] == 1
    assert report["class_counts_valid"]["monkeypox"] == 1
    assert report["class_counts_valid"]["non_monkeypox"] == 1
    assert report["class_counts_corrupt"]["monkeypox"] == 1
    assert report["valid_images"] == 2
    assert report["corrupt_count"] == 1
    assert report["total_files"] == 3
    assert report["imbalance_ratio"] == pytest.approx(1.0)
    assert len(report["corrupt_files"]) == 1
    assert report["corrupt_files"][0]["path"].endswith("broken.png")
    assert report["corrupt_files"][0]["error"]

    for item in report["files"]:
        assert "sha256" in item
        assert "bytes" in item
        if item["valid"]:
            assert item["width"] == 32
            assert item["height"] == 32
            assert item["format"] == "PNG"
            assert item["mode"] == "RGB"
            assert item["ahash"]

    json_path = metrics / "audit_report.json"
    txt_path = metrics / "audit_report.txt"
    assert json_path.is_file()
    assert txt_path.is_file()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["corrupt_count"] == 1
    assert "Corrupt files" in txt_path.read_text(encoding="utf-8")


def test_audit_sha_duplicates(tmp_path: Path) -> None:
    raw, metrics, root = _layout(tmp_path)
    original = raw / "monkeypox" / "orig.png"
    _write_pattern_png(original, "left")
    shutil.copy(original, raw / "monkeypox" / "copy.png")
    _write_pattern_png(raw / "non_monkeypox" / "other.png", "top")

    report = run_audit(data_raw=raw, output_dir=metrics, root=root)

    assert report["valid_images"] == 3
    assert report["class_counts"]["monkeypox"] == 2
    assert len(report["exact_duplicates"]) == 1
    group = report["exact_duplicates"][0]
    assert group["count"] == 2
    assert group["cross_class"] is False
    paths = {Path(p).name for p in group["paths"]}
    assert paths == {"orig.png", "copy.png"}
    member_shas = {
        item["sha256"]
        for item in report["files"]
        if Path(item["path"]).name in {"orig.png", "copy.png"}
    }
    assert len(member_shas) == 1
    assert group["sha256"] in member_shas


def test_audit_near_duplicates_resized_copy(tmp_path: Path) -> None:
    raw, metrics, root = _layout(tmp_path)
    _write_pattern_png(raw / "monkeypox" / "small.png", "left", size=(32, 32))
    _write_pattern_png(raw / "monkeypox" / "large.png", "left", size=(64, 64))
    _write_pattern_png(raw / "non_monkeypox" / "other.png", "top")

    report = run_audit(data_raw=raw, output_dir=metrics, root=root)

    shas = {
        item["sha256"]
        for item in report["files"]
        if Path(item["path"]).name in {"small.png", "large.png"}
    }
    assert len(shas) == 2
    names_in_near = [
        {Path(p).name for p in group["paths"]}
        for group in report["near_duplicates"]
    ]
    assert any({"small.png", "large.png"} <= names for names in names_in_near)


def test_audit_cross_class_identical_sha(tmp_path: Path) -> None:
    raw, metrics, root = _layout(tmp_path)
    src = raw / "monkeypox" / "shared.png"
    _write_pattern_png(src, "left")
    shutil.copy(src, raw / "non_monkeypox" / "shared.png")

    report = run_audit(data_raw=raw, output_dir=metrics, root=root)

    assert report["status"] == "issues"
    assert len(report["cross_class_exact"]) == 1
    group = report["cross_class_exact"][0]
    assert group["cross_class"] is True
    assert set(group["labels"]) == {"monkeypox", "non_monkeypox"}
    text = (metrics / "audit_report.txt").read_text(encoding="utf-8")
    assert "identical files appear in both classes" in text


def test_audit_empty_dataset(tmp_path: Path) -> None:
    raw, metrics, root = _layout(tmp_path)

    report = run_audit(data_raw=raw, output_dir=metrics, root=root)

    assert report["status"] == "empty"
    assert report["class_counts"] == {"non_monkeypox": 0, "monkeypox": 0}
    assert report["total_files"] == 0
    assert report["valid_images"] == 0
    assert report["files"] == []
    assert report["imbalance_ratio"] is None

    payload = json.loads((metrics / "audit_report.json").read_text(encoding="utf-8"))
    assert payload["status"] == "empty"
    text = (metrics / "audit_report.txt").read_text(encoding="utf-8")
    assert "data/raw/monkeypox" in text
    assert "data/raw/non_monkeypox" in text


def test_audit_empty_missing_dirs(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    metrics = tmp_path / "metrics"
    report = run_audit(data_raw=raw, output_dir=metrics, root=tmp_path)
    assert report["status"] == "empty"
    assert (metrics / "audit_report.json").is_file()


def test_main_empty_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw, metrics, _root = _layout(tmp_path)
    code = main(["--data-raw", str(raw), "--output-dir", str(metrics)])
    assert code == 2
    err = capsys.readouterr().err
    assert "No images found" in err
    assert "data/raw/monkeypox" in err
    assert EMPTY_STRUCTURE_MESSAGE.strip() in err
    assert json.loads((metrics / "audit_report.json").read_text())["status"] == "empty"


def test_main_ok_exits_0(tmp_path: Path) -> None:
    raw, metrics, _root = _layout(tmp_path)
    _write_pattern_png(raw / "monkeypox" / "a.png", "left")
    _write_pattern_png(raw / "non_monkeypox" / "b.png", "top")
    code = main(["--data-raw", str(raw), "--output-dir", str(metrics)])
    assert code == 0


def test_module_cli_empty(tmp_path: Path) -> None:
    raw, metrics, _root = _layout(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.src.audit",
            "--data-raw",
            str(raw),
            "--output-dir",
            str(metrics),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "No images found" in result.stderr
    assert (metrics / "audit_report.json").is_file()


def test_ignores_disallowed_extensions(tmp_path: Path) -> None:
    raw, metrics, root = _layout(tmp_path)
    _write_pattern_png(raw / "monkeypox" / "ok.png", "left")
    (raw / "monkeypox" / "notes.txt").write_text("not an image", encoding="utf-8")
    _write_pattern_png(raw / "non_monkeypox" / "ok.png", "top")

    report = run_audit(data_raw=raw, output_dir=metrics, root=root)
    assert report["total_files"] == 2
    assert all(item["path"].endswith(".png") for item in report["files"])


def test_average_hash_identical_and_distinct() -> None:
    left = Image.new("RGB", (32, 32), (0, 0, 0))
    ImageDraw.Draw(left).rectangle([0, 0, 16, 32], fill=(255, 255, 255))
    clone = left.copy()
    top = Image.new("RGB", (32, 32), (0, 0, 0))
    ImageDraw.Draw(top).rectangle([0, 0, 32, 16], fill=(255, 255, 255))

    assert average_hash(left) == average_hash(clone)
    assert hamming_distance(average_hash(left), average_hash(clone)) == 0
    assert hamming_distance(average_hash(left), average_hash(top)) > NEAR_DUPLICATE_HAMMING
    assert len(average_hash_hex(left)) == 16


def test_cluster_by_hamming_groups_near_duplicates() -> None:
    base = 0b1111_1111_1111_1111
    near = 0b1111_1111_1111_1000  # 3 bits differ
    far = 0b0000_0000_0000_0000
    clusters = cluster_by_hamming(
        {"a": base, "b": near, "c": far},
        threshold=5,
    )
    assert clusters == [["a", "b"]]
