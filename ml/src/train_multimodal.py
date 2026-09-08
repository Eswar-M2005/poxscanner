"""Two-stage training pipeline for the multimodal (image + clinical) model.

Stage 1: Image backbone is frozen. Only clinical encoder, projection, and fusion layers are trained.
Stage 2: Deeper blocks of the image backbone are unfrozen with a lower learning rate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.src.architectures.multimodal import MultimodalClassifier, build_multimodal_model
from ml.src.clinical import ClinicalPreprocessor
from ml.src.dataset import MultimodalLesionDataset
from ml.src.metrics import binary_metrics, dump_metrics_json, metrics_payload, save_curves
from ml.src.predict import get_device
from ml.src.utils import get_logger, load_yaml, repo_root, set_seed

log = get_logger("ml.src.train_multimodal")


def train_multimodal(
    config: dict[str, Any] | None = None,
    epochs_stage1: int = 5,
    epochs_stage2: int = 5,
    batch_size: int = 16,
    lr_head: float = 1e-3,
    lr_backbone: float = 1e-4,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
    root: Path | None = None,
    checkpoint_out: Path | str | None = None,
    max_train_samples: int | None = None,
) -> tuple[MultimodalClassifier, dict[str, Any], dict[str, Any]]:
    root = root or repo_root()
    device = device or get_device()
    set_seed(42)

    # 1. Fit clinical preprocessor strictly on training split
    splits_dir = root / "data" / "splits"
    clinical_file = root / "Data.csv"
    preprocessor = ClinicalPreprocessor(max_features=28)
    import pandas as pd

    c_df = pd.read_csv(clinical_file)
    preprocessor.fit(c_df, y=c_df["Status"])
    preprocessor.save(root / "ml" / "artifacts" / "metrics" / "clinical_features.json")

    # 2. Datasets & Loaders
    train_ds = MultimodalLesionDataset(
        split="train",
        splits_dir=splits_dir,
        clinical_path=clinical_file,
        clinical_preprocessor=preprocessor,
    )
    if max_train_samples is not None and max_train_samples < len(train_ds):
        train_ds.frame = train_ds.frame.sample(n=max_train_samples, random_state=42).reset_index(drop=True)

    val_ds = MultimodalLesionDataset(
        split="val",
        splits_dir=splits_dir,
        clinical_path=clinical_file,
        clinical_preprocessor=preprocessor,
    )
    test_ds = MultimodalLesionDataset(
        split="test",
        splits_dir=splits_dir,
        clinical_path=clinical_file,
        clinical_preprocessor=preprocessor,
    )

    num_workers = 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Calculate class weights for imbalanced loss
    train_labels = [row["label"] for _, row in train_ds.frame.iterrows()]
    pos_count = sum(1 for l in train_labels if l == "monkeypox")
    neg_count = len(train_labels) - pos_count
    pos_weight = float(neg_count / max(pos_count, 1))
    weights = torch.tensor([1.0, pos_weight], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # 3. Model setup
    clinical_dim = len(preprocessor.selected_features_)
    local_resnet = root / "ml" / "artifacts" / "checkpoints" / "resnet50_best.pt"
    if not local_resnet.is_file():
        local_resnet = root / "ml" / "artifacts" / "checkpoints" / "selected.pt"
    backbone_path = str(local_resnet) if local_resnet.is_file() else None

    model = build_multimodal_model(
        backbone_name="resnet50",
        clinical_dim=clinical_dim,
        num_classes=2,
        pretrained=True,
        pretrained_backbone_path=backbone_path,
        dropout=0.3,
    )
    model.to(device)

    # Helper evaluation function
    def evaluate_loader(loader: DataLoader) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        model.eval()
        y_true_list, y_prob_list = [], []
        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(device)
                clins = batch["clinical"].to(device)
                labels = batch["label"].to(device)
                logits = model(imgs, clins)
                probs = torch.softmax(logits, dim=1)[:, 1]
                y_true_list.append(labels.cpu().numpy())
                y_prob_list.append(probs.cpu().numpy())
        y_true = np.concatenate(y_true_list)
        y_prob = np.concatenate(y_prob_list)
        metrics = binary_metrics(y_true, y_prob)
        return metrics, y_true, y_prob

    # Stage 1: Freeze image backbone
    log.info("Starting Stage 1: Training clinical branch and fusion head (backbone frozen)")
    model.freeze_image_backbone()
    opt1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr_head,
        weight_decay=weight_decay,
    )

    best_val_f1 = -1.0
    best_state: dict[str, Any] = {}

    for epoch in range(1, epochs_stage1 + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Stage 1 Epoch {epoch}/{epochs_stage1}", leave=False):
            imgs = batch["image"].to(device)
            clins = batch["clinical"].to(device)
            labels = batch["label"].to(device)
            opt1.zero_grad()
            logits = model(imgs, clins)
            loss = criterion(logits, labels)
            loss.backward()
            opt1.step()
            losses.append(loss.item())

        val_metrics, _, _ = evaluate_loader(val_loader)
        f1 = val_metrics.get("f1", 0.0)
        log.info(
            "Stage 1 Epoch %d: Train Loss %.4f | Val F1 %.4f | Val AUC %.4f",
            epoch,
            float(np.mean(losses)),
            f1,
            val_metrics.get("roc_auc", 0.0),
        )
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Stage 2: Unfreeze deeper backbone blocks
    log.info("Starting Stage 2: Fine-tuning image backbone and fusion layers")
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.unfreeze_last_blocks()

    head_params = list(model.clinical_encoder.parameters()) + list(model.fusion_head.parameters())
    backbone_params = [p for p in model.image_backbone.parameters() if p.requires_grad]

    opt2 = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head * 0.5},
        ],
        weight_decay=weight_decay,
    )

    for epoch in range(1, epochs_stage2 + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Stage 2 Epoch {epoch}/{epochs_stage2}", leave=False):
            imgs = batch["image"].to(device)
            clins = batch["clinical"].to(device)
            labels = batch["label"].to(device)
            opt2.zero_grad()
            logits = model(imgs, clins)
            loss = criterion(logits, labels)
            loss.backward()
            opt2.step()
            losses.append(loss.item())

        val_metrics, _, _ = evaluate_loader(val_loader)
        f1 = val_metrics.get("f1", 0.0)
        log.info(
            "Stage 2 Epoch %d: Train Loss %.4f | Val F1 %.4f | Val AUC %.4f",
            epoch,
            float(np.mean(losses)),
            f1,
            val_metrics.get("roc_auc", 0.0),
        )
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Load best checkpoint
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Evaluate on test set
    test_metrics, y_true_test, y_prob_test = evaluate_loader(test_loader)
    log.info(
        "Final Test Evaluation: Accuracy %.4f | Macro F1 %.4f | Recall %.4f | ROC-AUC %.4f",
        test_metrics.get("accuracy", 0.0),
        test_metrics.get("f1_macro", test_metrics.get("f1", 0.0)),
        test_metrics.get("recall", 0.0),
        test_metrics.get("roc_auc", 0.0),
    )

    # Save checkpoint
    ckpt_path = Path(checkpoint_out) if checkpoint_out else root / "ml" / "artifacts" / "checkpoints" / "multimodal_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": "multimodal_resnet50",
        "state_dict": model.state_dict(),
        "backbone_name": "resnet50",
        "clinical_dim": clinical_dim,
        "selected_features": preprocessor.selected_features_,
        "metrics": test_metrics,
    }
    torch.save(payload, ckpt_path)
    log.info("Saved best multimodal checkpoint to %s", ckpt_path)

    metrics_dest = root / "ml" / "artifacts" / "metrics" / "multimodal_test.json"
    dump_metrics_json(metrics_dest, metrics_payload("multimodal_resnet50", "test", test_metrics, len(test_ds)))

    return model, val_metrics, test_metrics


if __name__ == "__main__":
    train_multimodal(epochs_stage1=2, epochs_stage2=2)
