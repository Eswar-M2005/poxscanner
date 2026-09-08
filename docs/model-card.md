# Model card

Format follows Mitchell et al., “Model Cards for Model Reporting” (FAccT 2019). **No trained weights are selected.** Status: `pending_training` in `ml/artifacts/metrics/selection.json`. Quantitative performance is **PENDING REAL TRAINING RESULTS**. Do not treat placeholder JSON nulls, demo-mode probabilities, or literature figures as this model’s accuracy, AUC, or F1.

The **dataset and split manifests exist** on this working copy (see [dataset.md](dataset.md) and `data/splits/split_meta.json`). Training/evaluation JSON has not been filled with completed experiment scores.

## Model details

| Field | Value |
| --- | --- |
| Name | Pox Scanner lesion classifier |
| Selected model | `pending_training` (`selected_model: null`) |
| Version | `0.1.0` |
| Task | Binary skin-lesion image classification |
| Classes | `non_monkeypox` (0), `monkeypox` (1) |
| Input | RGB image, 224×224, ImageNet normalization |
| Candidate architectures | EfficientNet-B0, DenseNet-121, ResNet-50, MobileNetV3-Small (ImageNet-pretrained, new 2-way head) |
| Framework | PyTorch (training); FastAPI serving |
| Checkpoint | None. Expected path after selection: `ml/artifacts/checkpoints/selected.pt` or `{model}_best.pt` |
| Person/org | Research / college codebase in this repository |

Until `python -m ml.scripts.compare_models` completes on real metrics, API `selection_status` is `pending_training` and demo mode may serve stub outputs.

## Intended use

**Primary intended uses**

- Educational demonstration of a leakage-aware CNN pipeline and Grad-CAM overlay on lesion photos.
- Research experimentation on the audited local set under `data/raw/` and the generated split CSVs.

**Primary intended users**

- Students and researchers working on this repository.
- Not patients, not clinicians as a diagnostic aid, not public health case-counting systems.

**Out-of-scope uses**

- Medical diagnosis, screening, triage, isolation decisions, or treatment choice.
- Symptom-based prediction (no tabular clinical features are ingested).
- Multi-class differential (chickenpox, measles, HSV, syphilis, HFMD, etc. are not separate heads; they are folded into `non_monkeypox` if present).
- Video, dermoscopy-specific devices, or whole-body montages unless resized as a single 224² crop (unsupported clinically).
- Deployment as a regulated medical device.

## Factors

Relevant factors (unevaluated; no disaggregated scores):

- Skin phototype / pigmentation
- Lesion stage (macule, vesicle, pustule, crust)
- Body site, lighting, camera, compression, crop tightness
- Geography and clade (historically endemic vs 2022+ outbreak presentations)
- Label quality (PCR vs clinical vs web caption)
- Filename family inside the negative class (HFMD, CHP, CWP, MSL, HEALTHY, …)

Disaggregated metrics by these factors **do not exist**.

## Metrics

Planned (see [methodology.md](methodology.md)): accuracy, precision, recall/sensitivity (positive = monkeypox), specificity, F1, ROC-AUC, PR-AUC, confusion matrix `[[TN, FP], [FN, TP]]`, p50 latency.

**Current values: PENDING REAL TRAINING RESULTS.** Placeholder files under `ml/artifacts/metrics/*_test.json` have `n_samples: 0` and null metric fields. An older `notes` string may still say the dataset was empty; that is stale. Splits currently contain 6374 / 272 / 270 images (test positives = 52).

Decision threshold: 0.5 unless a later evaluation protocol changes it. Selection priority: monkeypox recall, then F1, then AUCs, then specificity, then val–test F1 gap, then latency.

## Evaluation data

**Held-out split exists; scores do not.** Evaluation must use `data/splits/test.csv` only (270 images: 52 monkeypox, 218 non_monkeypox), with the same leakage rules as training. Do not evaluate on `data/raw` without splits.

From `split_meta.json`: 7518 enumerated, 602 exact duplicates removed, 6916 unique, 127 groups, seed 42.

## Training data

**Present in `data/raw/` and listed in `data/splits/train.csv` / `val.csv`.** Counts after exact-duplicate removal: train 6374 (2502 / 3872), val 272 (52 / 220). Audit: 2828 valid monkeypox files, 4690 valid non_monkeypox, all JPEG, 0 corrupt, imbalance ratio ~1.66. Near-duplicates: 849 aHash groups; 23 cross-class near-duplicate groups flagged for review.

Licence, confirmation standard, and demographic coverage remain **incompletely documented**. Treat the collection as a research/educational set. Provenance notes: [dataset.md](dataset.md).

This card will need a cited licence and labelling protocol before any claim beyond educational use.

## Quantitative analyses

**PENDING REAL TRAINING RESULTS.** No unitary or intersectional slices. No calibration plots. No trained-model latency on a defined device.

Demo-mode probabilities are a **hash of image bytes**, not a model.

## Ethical considerations

- Outputs must not be presented as a diagnosis. Disclaimer is mandatory in the API and UI.
- Skin photographs are sensitive. Default: do not persist uploads (`STORE_UPLOADS=false`).
- Dataset bias (typical web images, under-representation of darker skin and atypical lesions) will, if unmeasured, produce unequal error rates.
- Image-only input ignores systemic symptoms that Setegn & Dejene treat as the clinically meaningful feature set.
- Grad-CAM is not a causal explanation and can highlight irrelevant correlated pixels (background, rulers, stickers).
- Small test *n* for the positive class (52) will make any future single-number accuracy easy to over-interpret.

Full discussion: [ethics-and-limitations.md](ethics-and-limitations.md).

## Caveats and recommendations

1. Finish train → evaluate → `compare_models` on the existing splits before claiming any performance. **PENDING REAL TRAINING RESULTS** until then.
2. Report sensitivity and specificity with sample sizes; do not lead with accuracy.
3. Keep demo mode visually distinct from checkpoint inference.
4. Do not compare this image model’s future scores to the BMC 2025 symptom-model accuracy; the tasks differ.
5. Laboratory confirmation (e.g. PCR) remains the reference standard for mpox; this tool does not replace it.
6. Review the 23 cross-class near-duplicate groups before treating test error as pure generalization.
