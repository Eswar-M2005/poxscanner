"""Create train/val/test split manifests.

Usage (from repo root)::

    python -m ml.scripts.make_splits
    python -m ml.scripts.make_splits --raw-dir data/raw --out-dir data/splits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from ml.src.utils import repo_root
except ImportError:

    def repo_root() -> Path:
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "docs" / "contracts.md").is_file() or (
                parent / "pyproject.toml"
            ).is_file():
                return parent
        return here.parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build stratified 70/15/15 split CSVs from data/raw."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Class folders monkeypox/ and non_monkeypox/ (default: data/raw)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write train.csv/val.csv/test.csv (default: data/splits)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    raw_dir = args.raw_dir if args.raw_dir is not None else root / "data" / "raw"
    out_dir = args.out_dir if args.out_dir is not None else root / "data" / "splits"

    from ml.src.dataset import EmptyRawDatasetError, make_splits

    try:
        make_splits(
            raw_dir,
            out_dir,
            seed=args.seed,
            train=args.train,
            val=args.val,
            test=args.test,
        )
    except EmptyRawDatasetError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
