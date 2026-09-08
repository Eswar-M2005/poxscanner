# Methodology

Binary image classification of a single lesion photograph: `non_monkeypox` (0) vs `monkeypox` (1).

**Dataset and splits are present** on this working copy (`data/raw/` populated; `data/splits/{train,val,test}.csv` and `split_meta.json` written). **Completed experiment metrics are not.** Status: **PENDING REAL TRAINING RESULTS.** `ml/artifacts/metrics/selection.json` is `pending_training` (`selected_model: null`). Placeholder `*_test.json` files still contain null scores (`n_samples: 0`); those nulls are not accuracies. Do not invent AUC, F1, or accuracy numbers.

## Contrast with the literature baseline

Setegn & Dejene (*BMC Infectious Diseases* 2025;25:419) train **tabular** ensembles (Random Forest, boosting family, LGBMClassifier) on **clinical symptoms** and apply XAI (SHAP, LIME, ELI5, counterfactuals, QLattice). In related-work they argue that image-based monkeypox detectors “may not be effective in many real-life situations” and that several prior imaging studies “lacked explainability and interpretability.”

This repository cannot reproduce their symptom pipeline: the available data modality is images. We therefore:

- treat their result as a **literature baseline and a warning**, not a number to match;
- restrict the task to image-only binary classification and state that limitation in the product and ethics docs;
- require Grad-CAM overlays at inference (localization, not feature-importance over symptoms);
- require leakage control and metrics beyond accuracy, which imaging papers often omit.

Their reported LGBM accuracy (89.3% on a small, SMOTE-expanded symptom table) is **not** a target or a result of this codebase.

## Dataset used for experiments (local audit)

Audit: `python -m ml.src.audit` → `ml/artifacts/metrics/audit_report.json`. Splits: `python -m ml.scripts.make_splits` → `data/splits/split_meta.json`. Details: [dataset.md](dataset.md).

| Item | Value |
| --- | ---: |
| Audited files | 7518 (JPEG, 0 corrupt) |
| Exact SHA-256 duplicate files removed | 602 |
| Unique images kept | 6916 |
| Groups (`group_id`) | 127 |
| Train / val / test images | 6374 / 272 / 270 |
| Train (mpox / non) | 2502 / 3872 |
| Val (mpox / non) | 52 / 220 |
| Test (mpox / non) | 52 / 218 |
| Near-duplicate groups (aHash Hamming ≤ 5) | 849 |
| Cross-class near-duplicate groups | 23 (flagged; no cross-class exact SHA-256) |
| Seed | 42 |
| `used_metadata_csv` | false |
| `used_near_duplicates` | true |

Val/test are smaller than a naive 15% of 6916 images because splits are **group-level**. Interpret future metrics with *n* = 52 positive val/test cases. Training reads split CSVs only; `ml.src.train` still refuses if CSVs are missing or empty (e.g. a fresh clone without data).

## Models

ImageNet-pretrained backbones, two-class heads, dropout 0.2:

| Config | Backbone |
| --- | --- |
| `ml/config/efficientnet_b0.yaml` | EfficientNet-B0 |
| `ml/config/densenet121.yaml` | DenseNet-121 |
| `ml/config/resnet50.yaml` | ResNet-50 |
| `ml/config/mobilenet_v3.yaml` | MobileNetV3-Small |

`pretrained: true` by default. Heads replace the ImageNet classifier with `Dropout → Linear(num_classes=2)` (MobileNetV3 keeps a small two-layer head: Linear → Hardswish → Dropout → Linear). Architectures live under `ml/src/architectures/`; `ml/src/models.py` is a registry only.

## Transfer learning: freeze then fine-tune last blocks

Default schedule (`ml/config/*.yaml`, `ml/src/train.py`):

1. **Freeze backbone** for `freeze_epochs: 5`. Train the classification head at `lr_head: 0.001` (AdamW, `weight_decay: 0.0001`, cross-entropy).
2. **Unfreeze last backbone stages only** (`unfreeze_last_blocks`): ResNet `layer4`; DenseNet `denseblock4` (+ `norm5` if present); EfficientNet-B0 / MobileNetV3-Small last two `features` blocks. Fine-tune those at `lr_backbone: 0.0001`; head remains at `lr_head`.
3. Up to `epochs: 30`, `batch_size: 16`.
4. Early stopping: patience 7 on **`val_recall_monkeypox`** (positive class), not accuracy. Best weights: `ml/artifacts/checkpoints/{model.name}_best.pt`.

Rationale: small lesion sets overfit quickly if the full CNN is trained from epoch 1; freezing the encoder first fits a linear-ish head on frozen features, then a lower backbone LR adapts high-level filters without erasing ImageNet structure at once.

Mixed precision default: off. Seed: `42` (Python, NumPy, PyTorch; `cudnn.deterministic = True`, `cudnn.benchmark = False`).

**PENDING REAL TRAINING RESULTS** — this schedule is implemented; reported epoch curves and val recall are not yet available in metrics JSON.

## Transforms (conservative)

**Train only:** resize 224; horizontal flip \(p=0.5\); rotation ±10°; random resized crop scale 0.85–1.0; brightness/contrast jitter 0.1.

**Val / test / inference:** resize 224, center crop 224, ImageNet mean/std.

**Not used:** vertical flip, elastic deform, heavy color jitter. Aggressive color/geometry aug can invent lesion appearances that do not occur clinically and inflate train metrics.

## Leakage control

See [dataset.md](dataset.md). Training must use split CSVs only. Identical SHA-256 files are collapsed to one row. pHash-near-duplicate groups share a `group_id` and stay in one split. Filename prefixes (e.g. `MKP_17_*`) define patient/session groups when `metadata.csv` is absent. Image-level random splits are disallowed.

`ml.src.train` refuses to run if splits are missing or empty.

## Metrics beyond accuracy

Positive class = monkeypox. Confusion matrix layout `[[TN, FP], [FN, TP]]`.

Reported (when a real evaluation exists): accuracy, precision, recall, **sensitivity** (same as recall on the positive class), **specificity**, F1, **ROC-AUC**, **PR-AUC**, p50 latency (ms).

Accuracy alone is insufficient under class imbalance and when missing monkeypox (false negatives) is costlier than extra false positives. Sensitivity is the early-stopping key. Specificity is a guard against a trivial “always monkeypox” policy. ROC-AUC and PR-AUC evaluate ranking, not a single 0.5 threshold. Threshold default for discrete metrics: 0.5 unless a later protocol documents otherwise.

Placeholder metrics JSON uses `null` fields. Older notes may still say `Pending training — no dataset.` — that note is stale relative to the current local files. The truthful status is: **dataset present, splits present, PENDING REAL TRAINING RESULTS.**

## Model selection policy

Script: `python -m ml.scripts.compare_models`. Writes `ml/artifacts/metrics/selection.json`.

Priority (in order):

1. Recall / sensitivity for monkeypox
2. F1
3. ROC-AUC, then PR-AUC
4. Specificity (must not collapse)
5. Smaller |val − test| gap on F1 (overfit penalty)
6. Inference latency

Until experiments complete (`accuracy` still null in every `*_test.json`):

```json
{
  "status": "pending_training",
  "selected_model": null,
  "checkpoint": null,
  "reason": "No completed experiment metrics. Add images under data/raw and run training.",
  "ranking": []
}
```

That object is the current on-disk selection. It is not a ranking of trained models.

## Explainability at inference

Grad-CAM on the backbone target layer (`get_gradcam_target_layer`). Overlay PNG is returned with the prediction when `explain=true` (default). This highlights spatial regions that influenced the class score. It does **not** identify symptoms, does **not** prove etiology, and is not comparable to the tabular XAI in Setegn & Dejene. Demo mode may substitute a placeholder heatmap and must set `explanation.note` accordingly.

## Inference preprocessing

Same as val/test: decode → RGB → resize/center-crop 224 → ImageNet normalize → softmax over two logits. Probability keys: `monkeypox`, `non_monkeypox`.
