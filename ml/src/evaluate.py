"""Evaluate a checkpoint on a split CSV.

CLI::

    python -m ml.src.evaluate --checkpoint PATH --split test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.src.dataset import LesionDataset
from ml.src.metrics import binary_metrics, dump_metrics_json, metrics_payload, save_curves
from ml.src.predict import get_device, load_checkpoint
from ml.src.transforms import build_transforms
from ml.src.utils import get_logger, repo_root

log = get_logger("ml.src.evaluate")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Pox Scanner checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to a *.pt checkpoint")
    parser.add_argument("--split", default="test", help="Split name: train, val, or test")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args(argv)


def _unpack_batch(batch) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch.get("image", batch.get("x")), batch.get("label", batch.get("y"))
    return batch[0], batch[1]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = repo_root()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    if not ckpt_path.is_file():
        print(
            f"No checkpoint at {ckpt_path}. Train a model first:\n"
            "  python -m ml.src.train --model efficientnet_b0",
            file=sys.stderr,
        )
        return 1

    device = get_device()
    model, payload = load_checkpoint(ckpt_path, device=device)
    config = payload.get("config") or {}
    model_name = payload.get("model_name") or (config.get("model") or {}).get("name")
    if not model_name:
        print("Checkpoint is missing model_name.", file=sys.stderr)
        return 1

    paths = config.get("paths") or {}
    splits_dir = root / paths.get("splits_dir", "data/splits")
    split = str(args.split).strip().lower()
    csv_path = splits_dir / f"{split}.csv"
    if not csv_path.is_file():
        print(
            f"Split CSV not found: {csv_path}\n"
            "Add images under data/raw and run:\n"
            "  python -m ml.src.audit\n"
            "  python -m ml.scripts.make_splits",
            file=sys.stderr,
        )
        return 1
    frame = pd.read_csv(csv_path)
    if frame.empty:
        print(f"Split CSV is empty: {csv_path}", file=sys.stderr)
        return 1

    train_cfg = config.get("train") or {}
    image_size = int(payload.get("image_size") or train_cfg.get("image_size") or 224)
    batch_size = int(args.batch_size or train_cfg.get("batch_size") or 16)
    transform_split = "val" if split == "train" else split
    if transform_split not in ("train", "val", "test", "inference"):
        transform_split = "val"
    dataset = LesionDataset(
        split=split,
        splits_dir=splits_dir,
        transform=build_transforms(transform_split, image_size=image_size),
        csv_path=csv_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if device.type != "cuda" else int(train_cfg.get("num_workers") or 0),
        pin_memory=device.type == "cuda",
    )

    model.eval()
    y_true_parts: list[np.ndarray] = []
    y_prob_parts: list[np.ndarray] = []
    latencies: list[float] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval {split}", leave=False):
            images, labels = _unpack_batch(batch)
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0 / max(int(images.size(0)), 1))
            probs = torch.softmax(logits, dim=1)[:, 1]
            y_true_parts.append(labels.cpu().numpy())
            y_prob_parts.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true_parts)
    y_prob = np.concatenate(y_prob_parts)
    metrics = binary_metrics(y_true, y_prob)
    latency = float(np.percentile(latencies, 50)) if latencies else None

    metrics_dir = root / paths.get("metrics_dir", "ml/artifacts/metrics")
    plots_dir = root / paths.get("plots_dir", "ml/artifacts/plots")
    dest = metrics_dir / f"{model_name}_{split}.json"
    payload_json = metrics_payload(
        model=str(model_name),
        split=split,
        metrics=metrics,
        n_samples=int(y_true.shape[0]),
        notes=f"Evaluated checkpoint {ckpt_path.name} on split={split}.",
        latency_ms_p50=latency,
    )
    dump_metrics_json(dest, payload_json)
    save_curves(
        y_true=y_true,
        y_prob=y_prob,
        confusion_matrix=metrics["confusion_matrix"],
        out_dir=plots_dir,
        prefix=f"{model_name}_{split}",
    )
    log.info("Wrote %s", dest)
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
