"""Two-phase transfer-learning trainer.

CLI::

    python -m ml.src.train --model efficientnet_b0
    python -m ml.src.train --config ml/config/efficientnet_b0.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ml.src.dataset import LABEL_TO_INDEX, LesionDataset
from ml.src.metrics import binary_metrics, dump_metrics_json, metrics_payload, save_curves
from ml.src.models import (
    build_model,
    freeze_backbone,
    head_parameters,
    trainable_backbone_parameters,
    unfreeze_last_blocks,
)
from ml.src.predict import get_device
from ml.src.transforms import build_transforms
from ml.src.utils import get_logger, load_yaml, repo_root, set_seed

VERSION = "0.1.0"
SPLITS_MISSING_MSG = """\
Splits are missing or empty. Add lesion images under:
  data/raw/monkeypox/
  data/raw/non_monkeypox/
then run:
  python -m ml.src.audit
  python -m ml.scripts.make_splits
Do not synthesize training images. Training will not create a fake dataset.
"""

log = get_logger("ml.src.train")


def _transforms(split: str, image_size: int):
    return build_transforms(split, image_size=image_size)


def _make_dataset(split: str, transform, splits_dir: Path, root: Path | None = None) -> Dataset:
    del root  # LesionDataset resolves repo-relative paths via repo_root()
    csv_path = splits_dir / f"{split}.csv"
    return LesionDataset(
        split=split,
        splits_dir=splits_dir,
        transform=transform,
        csv_path=csv_path,
    )


def _unpack_batch(batch) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        x = batch.get("image", batch.get("x"))
        y = batch.get("label", batch.get("y"))
        return x, y
    return batch[0], batch[1]


def _require_splits(root: Path, config: dict[str, Any]) -> Path:
    splits_dir = root / config["paths"]["splits_dir"]
    problems: list[str] = []
    for name in ("train", "val", "test"):
        path = splits_dir / f"{name}.csv"
        if not path.is_file():
            problems.append(f"missing {path}")
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            problems.append(f"unreadable {path}: {exc}")
            continue
        if frame.empty:
            problems.append(f"empty {path}")
    if problems:
        detail = "; ".join(problems)
        raise SystemExit(f"{SPLITS_MISSING_MSG}\nDetails: {detail}")
    return splits_dir


def _resolve_config_path(args: argparse.Namespace, root: Path) -> Path:
    if args.config:
        path = Path(args.config)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise SystemExit(f"Config not found: {path}")
        return path
    model_name = args.model or "efficientnet_b0"
    named = root / "ml" / "config" / f"{model_name}.yaml"
    if named.is_file():
        return named
    base = root / "ml" / "config" / "base.yaml"
    if base.is_file():
        return base
    raise SystemExit(f"No config for model {model_name!r} and base.yaml is missing.")


def _load_config(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    path = _resolve_config_path(args, root)
    config = load_yaml(path)
    if args.model:
        config.setdefault("model", {})
        if not args.config:
            config["model"]["name"] = args.model
        elif not config.get("model", {}).get("name"):
            config["model"]["name"] = args.model
    config.setdefault("model", {})
    config.setdefault("train", {})
    config.setdefault("paths", {})
    config["model"].setdefault("name", args.model or "efficientnet_b0")
    config["model"].setdefault("pretrained", True)
    config["model"].setdefault("num_classes", 2)
    config["model"].setdefault("dropout", 0.2)
    train = config["train"]
    train.setdefault("epochs", 30)
    train.setdefault("freeze_epochs", 5)
    train.setdefault("batch_size", 16)
    train.setdefault("num_workers", 2)
    train.setdefault("lr_head", 0.001)
    train.setdefault("lr_backbone", 0.0001)
    train.setdefault("weight_decay", 0.0001)
    train.setdefault("early_stopping_patience", 7)
    train.setdefault("early_stopping_metric", "val_recall_monkeypox")
    train.setdefault("seed", 42)
    train.setdefault("image_size", 224)
    train.setdefault("mixed_precision", False)
    paths = config["paths"]
    paths.setdefault("data_raw", "data/raw")
    paths.setdefault("splits_dir", "data/splits")
    paths.setdefault("checkpoint_dir", "ml/artifacts/checkpoints")
    paths.setdefault("metrics_dir", "ml/artifacts/metrics")
    paths.setdefault("plots_dir", "ml/artifacts/plots")
    return config


def _optimizer(model: nn.Module, name: str, lr_head: float, lr_backbone: float, weight_decay: float):
    head = [p for p in head_parameters(model, name) if p.requires_grad]
    backbone = trainable_backbone_parameters(model, name)
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": lr_backbone})
    if head:
        groups.append({"params": head, "lr": lr_head})
    if not groups:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr_head}]
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def _dataloader_kwargs(device: torch.device, num_workers: int) -> dict[str, Any]:
    workers = int(num_workers)
    if device.type != "cuda":
        workers = 0
    return {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any = None,
    use_amp: bool = False,
    desc: str = "",
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    y_true_parts: list[np.ndarray] = []
    y_prob_parts: list[np.ndarray] = []
    context = torch.enable_grad() if training else torch.no_grad()
    amp_device = device.type if device.type in ("cuda", "cpu") else "cpu"
    with context:
        iterator = tqdm(loader, desc=desc, leave=False)
        for batch in iterator:
            images, labels = _unpack_batch(batch)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            if use_amp and scaler is not None:
                with torch.autocast(device_type=amp_device):
                    logits = model(images)
                    loss = criterion(logits, labels)
                if training:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                if training:
                    loss.backward()
                    optimizer.step()
            losses.append(float(loss.item()))
            probs = torch.softmax(logits.detach(), dim=1)[:, 1]
            y_true_parts.append(labels.detach().cpu().numpy())
            y_prob_parts.append(probs.cpu().numpy())
            iterator.set_postfix(loss=float(np.mean(losses)))
    y_true = np.concatenate(y_true_parts) if y_true_parts else np.array([], dtype=int)
    y_prob = np.concatenate(y_prob_parts) if y_prob_parts else np.array([], dtype=np.float64)
    metrics = binary_metrics(y_true, y_prob)
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    metrics["loss"] = mean_loss
    return metrics, y_true, y_prob, mean_loss


def _write_split_metrics(
    root: Path,
    config: dict[str, Any],
    model_name: str,
    split: str,
    metrics: dict[str, Any],
    n_samples: int,
    notes: str = "",
    latency_ms_p50: float | None = None,
) -> Path:
    dest = root / config["paths"]["metrics_dir"] / f"{model_name}_{split}.json"
    payload = metrics_payload(
        model=model_name,
        split=split,
        metrics=metrics,
        n_samples=n_samples,
        notes=notes,
        latency_ms_p50=latency_ms_p50,
    )
    return dump_metrics_json(dest, payload)


def train(config: dict[str, Any], root: Path | None = None) -> Path:
    root = root or repo_root()
    train_cfg = config["train"]
    model_cfg = config["model"]
    name = str(model_cfg["name"])
    seed = int(train_cfg["seed"])
    set_seed(seed)
    splits_dir = _require_splits(root, config)

    device = get_device()
    log.info("Device: %s | model: %s | seed: %s", device, name, seed)

    image_size = int(train_cfg["image_size"])
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg["num_workers"])
    train_ds = _make_dataset("train", _transforms("train", image_size), splits_dir, root)
    val_ds = _make_dataset("val", _transforms("val", image_size), splits_dir, root)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit(SPLITS_MISSING_MSG)

    loader_kw = _dataloader_kwargs(device, num_workers)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        **loader_kw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kw,
    )

    model = build_model(
        name,
        num_classes=int(model_cfg.get("num_classes", 2)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    epochs = int(train_cfg["epochs"])
    freeze_epochs = int(train_cfg["freeze_epochs"])
    patience = int(train_cfg["early_stopping_patience"])
    lr_head = float(train_cfg["lr_head"])
    lr_backbone = float(train_cfg["lr_backbone"])
    weight_decay = float(train_cfg["weight_decay"])
    mixed_precision = bool(train_cfg.get("mixed_precision", False))
    use_amp = mixed_precision and device.type == "cuda"
    scaler = None
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except (TypeError, AttributeError):
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    ckpt_dir = root / config["paths"]["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{name}_best.pt"
    plots_dir = root / config["paths"]["plots_dir"]

    best_recall: float | None = None
    best_val_metrics: dict[str, Any] | None = None
    best_epoch = 0
    stale = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    last_train_metrics: dict[str, Any] = {}
    last_train_n = 0

    def _phase_optimizer(phase: int) -> torch.optim.Optimizer:
        if phase == 1:
            freeze_backbone(model, name)
            return _optimizer(model, name, lr_head=lr_head, lr_backbone=lr_head, weight_decay=weight_decay)
        unfreeze_last_blocks(model, name)
        return _optimizer(
            model,
            name,
            lr_head=lr_head,
            lr_backbone=lr_backbone,
            weight_decay=weight_decay,
        )

    optimizer = _phase_optimizer(1)
    current_phase = 1

    for epoch in range(1, epochs + 1):
        wanted_phase = 1 if epoch <= freeze_epochs else 2
        if freeze_epochs <= 0:
            wanted_phase = 2
        if wanted_phase != current_phase:
            log.info("Entering phase %s at epoch %s", wanted_phase, epoch)
            optimizer = _phase_optimizer(wanted_phase)
            current_phase = wanted_phase

        train_metrics, _, _, train_loss = _run_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            desc=f"train {epoch}/{epochs}",
        )
        val_metrics, _, _, val_loss = _run_epoch(
            model,
            val_loader,
            device,
            criterion,
            optimizer=None,
            desc=f"val {epoch}/{epochs}",
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        last_train_metrics = train_metrics
        last_train_n = int(len(train_ds))

        recall = val_metrics.get("recall")
        recall_f = -1.0 if recall is None else float(recall)
        log.info(
            "epoch %s phase %s train_loss=%.4f val_loss=%.4f val_recall_monkeypox=%.4f",
            epoch,
            current_phase,
            train_loss,
            val_loss,
            recall_f,
        )

        if best_recall is None or recall_f > best_recall:
            best_recall = recall_f
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            stale = 0
            payload = {
                "model_name": name,
                "state_dict": model.state_dict(),
                "config": config,
                "val_metrics": best_val_metrics,
                "epoch": epoch,
                "seed": seed,
                "class_to_idx": dict(LABEL_TO_INDEX),
                "image_size": image_size,
                "version": VERSION,
            }
            torch.save(payload, ckpt_path)
            log.info("Saved best checkpoint to %s (val_recall=%.4f)", ckpt_path, best_recall)
        else:
            stale += 1
            if stale >= patience:
                log.info("Early stopping at epoch %s (patience=%s)", epoch, patience)
                break

    recall_note = "n/a" if best_recall is None else f"{best_recall:.4f}"
    notes = (
        f"Two-phase transfer learning. Best epoch {best_epoch} "
        f"val_recall_monkeypox={recall_note}."
    )
    _write_split_metrics(root, config, name, "train", last_train_metrics, last_train_n, notes=notes)
    val_to_write = best_val_metrics or {}
    _write_split_metrics(root, config, name, "val", val_to_write, int(len(val_ds)), notes=notes)
    save_curves(
        history=history,
        out_dir=plots_dir,
        prefix=f"{name}_train",
    )
    if best_val_metrics and best_val_metrics.get("confusion_matrix") is not None:
        save_curves(
            confusion_matrix=best_val_metrics["confusion_matrix"],
            out_dir=plots_dir,
            prefix=f"{name}_val",
        )
    log.info("Training finished. Best checkpoint: %s", ckpt_path)
    return ckpt_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Pox Scanner classifier")
    parser.add_argument("--model", default=None, help="Registered model name")
    parser.add_argument("--config", default=None, help="Path to a YAML config")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.model is None and args.config is None:
        print("Provide --model NAME or --config PATH", file=sys.stderr)
        return 2
    root = repo_root()
    config = _load_config(args, root)
    train(config, root=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
