"""Three-model benchmark & ablation comparison:

Model A — Image Only (ResNet-50 CNN)
Model B — Clinical Only (Clinical MLP)
Model C — Multimodal (ResNet-50 + Clinical MLP Fusion)

Evaluates on the test split and outputs a comprehensive comparison matrix.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.src.architectures.multimodal import (
    ClinicalOnlyClassifier,
    build_clinical_model,
    build_multimodal_model,
)
from ml.src.clinical import ClinicalPreprocessor
from ml.src.dataset import MultimodalLesionDataset
from ml.src.metrics import binary_metrics
from ml.src.predict import get_device, load_checkpoint
from ml.src.train_multimodal import train_multimodal
from ml.src.utils import get_logger, repo_root, set_seed

log = get_logger("ml.src.compare_models")


def compute_extended_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    """Compute comprehensive diagnostic evaluation metrics."""
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    # Per-class metrics
    class_0_p = float(precision_score(y_true == 0, y_pred == 0, zero_division=0))
    class_0_r = float(recall_score(y_true == 0, y_pred == 0, zero_division=0))
    class_0_f = float(f1_score(y_true == 0, y_pred == 0, zero_division=0))

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "sensitivity": rec,
        "specificity": spec,
        "f1": f1,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            "non_monkeypox": {"precision": class_0_p, "recall": class_0_r, "f1": class_0_f},
            "monkeypox": {"precision": prec, "recall": rec, "f1": f1},
        },
    }


def train_clinical_only_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    clinical_dim: int,
    epochs: int = 15,
    device: torch.device | None = None,
) -> ClinicalOnlyClassifier:
    """Train Model B: Clinical-Only MLP baseline."""
    device = device or get_device()
    model = build_clinical_model(clinical_dim=clinical_dim)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_state: dict[str, Any] = {}

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            clins = batch["clinical"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(clins)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                clins = batch["clinical"].to(device)
                labels = batch["label"].to(device)
                logits = model(clins)
                loss = criterion(logits, labels)
                val_losses.append(loss.item())
        mean_vloss = float(np.mean(val_losses)) if val_losses else float("inf")
        if mean_vloss < best_loss:
            best_loss = mean_vloss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def compare_all_models(
    quick_run: bool = False,
    root: Path | None = None,
) -> dict[str, Any]:
    """Execute complete 3-model comparison and return results."""
    root = root or repo_root()
    device = get_device()
    set_seed(42)

    # 1. Setup preprocessor & loaders
    clinical_file = root / "Data.csv"
    preprocessor = ClinicalPreprocessor(max_features=28)
    c_df = pd.read_csv(clinical_file)
    preprocessor.fit(c_df, y=c_df["Status"])

    test_ds = MultimodalLesionDataset(
        split="test",
        clinical_path=clinical_file,
        clinical_preprocessor=preprocessor,
    )
    val_ds = MultimodalLesionDataset(
        split="val",
        clinical_path=clinical_file,
        clinical_preprocessor=preprocessor,
    )
    train_ds = MultimodalLesionDataset(
        split="train",
        clinical_path=clinical_file,
        clinical_preprocessor=preprocessor,
    )

    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    # ----------------------------------------------------
    # Model A: Image Only
    # ----------------------------------------------------
    log.info("Evaluating Model A: Image-Only Baseline")
    image_ckpt_path = root / "ml" / "artifacts" / "checkpoints" / "resnet50_best.pt"
    if not image_ckpt_path.is_file():
        image_ckpt_path = root / "ml" / "artifacts" / "checkpoints" / "selected.pt"

    img_model, _ = load_checkpoint(image_ckpt_path, device=device)
    img_model.eval()

    y_true_a, y_prob_a = [], []
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = img_model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1]
            y_true_a.append(labels.cpu().numpy())
            y_prob_a.append(probs.cpu().numpy())
    y_true_arr = np.concatenate(y_true_a)
    y_prob_a_arr = np.concatenate(y_prob_a)
    metrics_a = compute_extended_metrics(y_true_arr, y_prob_a_arr)

    # ----------------------------------------------------
    # Model B: Clinical Only
    # ----------------------------------------------------
    log.info("Training & Evaluating Model B: Clinical-Only Baseline")
    clin_dim = len(preprocessor.selected_features_)
    clin_epochs = 5 if quick_run else 20
    clin_model = train_clinical_only_model(
        train_loader=train_loader,
        val_loader=val_loader,
        clinical_dim=clin_dim,
        epochs=clin_epochs,
        device=device,
    )
    clin_model.eval()

    y_prob_b = []
    with torch.no_grad():
        for batch in test_loader:
            clins = batch["clinical"].to(device)
            logits = clin_model(clins)
            probs = torch.softmax(logits, dim=1)[:, 1]
            y_prob_b.append(probs.cpu().numpy())
    y_prob_b_arr = np.concatenate(y_prob_b)
    metrics_b = compute_extended_metrics(y_true_arr, y_prob_b_arr)

    # ----------------------------------------------------
    # Model C: Multimodal Fusion
    # ----------------------------------------------------
    log.info("Training & Evaluating Model C: Multimodal Fusion Network")
    s1 = 1 if quick_run else 3
    s2 = 1 if quick_run else 3
    multi_model, _, _ = train_multimodal(
        epochs_stage1=s1,
        epochs_stage2=s2,
        batch_size=16,
        device=device,
        root=root,
        max_train_samples=400 if quick_run else None,
    )
    multi_model.eval()

    y_prob_c = []
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            clins = batch["clinical"].to(device)
            logits = multi_model(imgs, clins)
            probs = torch.softmax(logits, dim=1)[:, 1]
            y_prob_c.append(probs.cpu().numpy())
    y_prob_c_arr = np.concatenate(y_prob_c)
    metrics_c = compute_extended_metrics(y_true_arr, y_prob_c_arr)

    # Aggregate results
    comparison = {
        "test_n_samples": int(len(y_true_arr)),
        "models": {
            "image_only": {
                "name": "Model A: Image-Only (ResNet-50)",
                "metrics": metrics_a,
            },
            "clinical_only": {
                "name": "Model B: Clinical-Only (MLP)",
                "metrics": metrics_b,
            },
            "multimodal": {
                "name": "Model C: Multimodal (ResNet-50 + Clinical MLP)",
                "metrics": metrics_c,
            },
        },
    }

    # Save to metrics directory
    dest = root / "ml" / "artifacts" / "metrics" / "model_comparison.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    log.info("Saved model comparison results to %s", dest)

    # Print summary table
    print("\n" + "=" * 80)
    print("                THREE-MODEL TEST SET COMPARISON MATRIX")
    print("=" * 80)
    print(f"{'Metric':<20} | {'Image-Only':<15} | {'Clinical-Only':<15} | {'Multimodal':<15}")
    print("-" * 80)
    for m in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "recall", "specificity", "precision", "roc_auc"]:
        val_a = metrics_a.get(m, 0.0)
        val_b = metrics_b.get(m, 0.0)
        val_c = metrics_c.get(m, 0.0)
        print(f"{m:<20} | {val_a:<15.4f} | {val_b:<15.4f} | {val_c:<15.4f}")
    print("=" * 80 + "\n")

    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer epochs")
    args = parser.parse_args()
    compare_all_models(quick_run=args.quick)
