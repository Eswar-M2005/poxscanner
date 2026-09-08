# Pox Scanner — College Project Report

**Title:** Leakage-aware transfer learning for binary mpox vs non-mpox lesion photographs, with Grad-CAM explanations and a non-diagnostic web demo.

**Version:** 0.1.0  
**Task:** Binary image classification — `non_monkeypox` (class 0) vs `monkeypox` (class 1).  
**Disclaimer:** Research and educational use only. This tool does not provide a medical diagnosis. Laboratory confirmation (for example PCR) remains the reference standard for mpox.

**Status of experiments:** **PENDING REAL TRAINING RESULTS.** No selected checkpoint. `ml/artifacts/metrics/selection.json` currently reports `status: pending_training`. Do not treat placeholder JSON nulls, demo-mode probabilities, or literature numbers as this project’s accuracy, AUC, or F1.

Companion documents: [dataset.md](dataset.md), [methodology.md](methodology.md), [model-card.md](model-card.md), [ethics-and-limitations.md](ethics-and-limitations.md), [architecture.md](architecture.md), [contracts.md](contracts.md).

---

## 1. Problem Statement

Mpox (monkeypox) can present as cutaneous lesions that look similar to other dermatoses (chickenpox, hand-foot-and-mouth disease, measles, and others). Public image collections have been used in research to train convolutional networks that map a single photograph to a binary label. Those studies often:

- split images randomly, so near-duplicate crops of the same lesion leak from train into test;
- report accuracy alone on imbalanced data;
- omit any spatial explanation of the prediction;
- are then described, incorrectly, as diagnostic tools.

Setegn and Dejene (*BMC Infectious Diseases*, 2025;25:419) train **tabular** ensembles on **clinical symptoms** and argue that image-only detectors may be ineffective in many real-life situations and that several imaging papers lacked explainability. Their LGBM accuracy (89.3% on a small, SMOTE-expanded symptom table) is **not** a result of this codebase and is **not** a target to match. This project cannot reproduce their symptom pipeline: the available modality is photographs.

**Pox Scanner** therefore states a narrower, honest problem:

> Given one RGB lesion photograph, estimate softmax probabilities for `{non_monkeypox, monkeypox}`, keep exact and near-duplicate images from crossing train/val/test, compare several ImageNet-pretrained CNNs with a recall-first selection policy, and return a Grad-CAM overlay so a student can see *where* the network looked — without claiming a diagnosis.

The product is an educational client–server demo. Demo mode (no checkpoint) returns **hash-based placeholder probabilities**, clearly flagged. That is not a model.

---

## 2. Objectives

1. **Define a binary, image-only task** with a fixed class encoding and a mandatory non-diagnostic disclaimer in the API and UI.
2. **Audit and clean** a local lesion-image collection under `data/raw/{monkeypox,non_monkeypox}/` (corrupt-file checks, format inventory, class counts).
3. **Remove exact duplicates** (SHA-256) and **group near-duplicates** (average hash, Hamming distance ≤ 5) so they cannot inflate test scores.
4. **Split by patient/session group**, not by independent images, targeting 70 / 15 / 15 at the *group* level (seed 42).
5. **Train** four ImageNet-pretrained backbones with a freeze-then-partial-unfreeze schedule, conservative augmentation, and early stopping on **validation recall for monkeypox**.
6. **Evaluate** with metrics beyond accuracy (sensitivity, specificity, F1, ROC-AUC, PR-AUC, confusion matrix, p50 latency) on the held-out test CSV only.
7. **Select** a single serving model with an explicit ranking policy (`python -m ml.scripts.compare_models`).
8. **Explain** predictions at inference with Grad-CAM overlays (localization, not causal proof).
9. **Serve** a FastAPI inference API and a Next.js UI that remain usable in demo mode until a real checkpoint exists.
10. **Document limitations and ethics** so the work cannot be mistaken for a medical device.

Objectives 5–7 are implemented in code. **Completed experimental scores are PENDING REAL TRAINING RESULTS.**

---

## 3. Dataset

Full layout and provenance notes: [dataset.md](dataset.md). Images are local and gitignored; a fresh checkout without `data/raw` must fail audit/split/train rather than invent files.

**Source (local, not auto-downloaded).** Files were copied into the contract folders `data/raw/monkeypox/` and `data/raw/non_monkeypox/`. Filenames such as `MKP_17_02_9.jpg`, `HFMD_46_01_2.jpg`, `CHP_01_01_1.jpg`, `CWP_03_01_1.jpg`, `MSL_05_02_5.jpg`, and `HEALTHY_84_01_7.jpg` indicate a mixture of mpox photos and other conditions (hand-foot-and-mouth, chickenpox, cowpox, measles-style labels, healthy skin) **folded into one negative class**. Licence, consent, geography, camera mix, and PCR vs clinical vs web labelling are **not fully documented** in this repository; the collection should be treated as a research/educational set, not a population sample.

Public datasets of this kind (for example MSLD / MSLD v2.0) are discussed in `scripts/download_msld_notes.md`. This report does not claim a specific public dump as the sole source beyond what the local audit shows.

### 3.1 Audit snapshot

From `ml/artifacts/metrics/audit_report.json` / `.txt` (`python -m ml.src.audit`, scanned 2026-08-28):

| Class | Files found | Valid | Corrupt |
| --- | ---: | ---: | ---: |
| `monkeypox` | 2828 | 2828 | 0 |
| `non_monkeypox` | 4690 | 4690 | 0 |
| **Total** | **7518** | **7518** | **0** |

- All files JPEG, RGB, typically already 224×224.
- Imbalance ratio (majority / minority valid counts): **1.66**.
- Audit status: `issues` because **23** near-duplicate groups span both classes (see Duplicate Handling). Zero cross-class *exact* SHA-256 matches.

### 3.2 Split snapshot

From `data/splits/split_meta.json` (`python -m ml.scripts.make_splits`, seed 42, `status: ok`):

| Quantity | Value |
| --- | ---: |
| Enumerated (audited) | **7518** |
| Exact SHA-256 duplicates removed | **602** |
| Unique images kept | **6916** |
| Patient/session groups | **127** |
| Train images | **6374** (monkeypox 2502, non_monkeypox 3872) |
| Val images | **272** (monkeypox 52, non_monkeypox 220) |
| Test images | **270** (monkeypox 52, non_monkeypox 218) |

Group-level target ratios are 0.70 / 0.15 / 0.15. Because each group holds many augmented copies (~14–20 files) and there are only 127 groups, **image counts are not a flat 70/15/15**. Val and test are smaller than a naive image-level split would give. That is intentional leakage control. Test monkeypox *n* = 52 implies **wide confidence intervals** once metrics exist; do not over-interpret a single percentage.

`used_metadata_csv`: false. All 7518 enumerated names contributed a filename-derived `group_id` (pattern `PREFIX_NUMBER`, e.g. `MKP_17`). Near-duplicate merging was applied (`used_near_duplicates: true`).

CSV columns: `path,label,split,sha256,width,height,group_id`. Training and evaluation **must** use these CSVs, not a raw folder walk.

---

## 4. Dataset Cleaning

Cleaning is deterministic and refuses to fabricate images.

1. **Layout.** Only `data/raw/monkeypox/` and `data/raw/non_monkeypox/` are scanned. Allowed extensions: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`. Hidden files are skipped.
2. **Decode.** Pillow opens each file; unreadable images are counted as corrupt and skipped at split time (`n_corrupt_skipped: 0` in the current `split_meta.json`).
3. **Identity.** SHA-256 of file bytes; width/height after decode; 8×8 average hash (aHash) for near-duplicate clustering.
4. **Class folders are the label.** There is no separate symptom table. Negative-class prefixes (HFMD, CHP, CWP, MSL, HEALTHY) are **not** extra heads; they are all `non_monkeypox`.
5. **No PHI in manifests.** `group_id` is an opaque split key (filename prefix or hash), not a medical record number. EXIF GPS is not copied into CSVs.
6. **Empty-data behaviour.** If `data/raw` is empty on another machine, `python -m ml.src.audit` exits non-zero; `make_splits` and `train` refuse to run. That guard remains even though this working copy is populated.

Images are **not** resized or colour-corrected as a stored “processed dataset.” Geometric and photometric transforms happen on the fly (Section 7). `data/processed/` is optional and unused for the numbers above.

---

## 5. Duplicate Handling

Lesion web sets republish the same photograph as `_ORIGINAL.jpg` plus numbered crops (`_1.jpg` … `_13.jpg`). Treating those as independent patients would leak identity into every split.

**Exact duplicates.** Files with the same SHA-256 are collapsed: one copy is kept, extras are dropped. The audit lists **602** exact-duplicate *groups*; split generation records **602** files removed (`n_exact_duplicates_removed`). Typical pattern: `*_13.jpg` identical to `*_ORIGINAL.jpg`. Cross-class exact matches: **0**.

**Near-duplicates.** Average hash (grayscale, 8×8, bit = 1 iff pixel > mean). Two hashes are near-duplicates if Hamming distance ≤ 5. Clustering is union-find (transitive). The audit reports **849** near-duplicate groups.

**Cross-class near-duplicates.** **23** groups contain paths in both `monkeypox` and `non_monkeypox`. Examples in the audit include perceptually similar crops of MKP vs HFMD/MSL/HEALTHY files. These are **flagged for manual review**. They did not create identical-byte label conflicts. Split grouping uses filename-derived patient/session ids **and** merges near-duplicate SHA-256 clusters into one `group_id`, so a near-duplicate pair cannot be assigned to two splits. They can still be a **label-noise** risk if two different diagnoses look similar under aHash; that is recorded as an audit `issues` status, not silently ignored.

**What is not done.** No SMOTE, no synthetic GAN images, no deletion of the 23 cross-class near-duplicate groups by default. Cleaning prefers honesty about remaining risk over an undocumented purge.

---

## 6. Leakage Prevention

Leakage here means the model seeing the *same lesion identity* in training and in val/test, which produces optimistic accuracy that will not hold on a new patient.

Rules implemented in `ml/src/dataset.py` (see also [dataset.md](dataset.md)):

1. Deduplicate exact SHA-256 (keep one file).
2. Merge near-duplicate clusters into a single `group_id`.
3. Prefer `data/raw/metadata.csv` `group_id` if present (not used in the current run).
4. Else derive `group_id` from the filename (`MKP_17_02_9` → `MKP_17`).
5. Else fall back to the file SHA-256.
6. **Assign splits at group level**, stratified by the group’s majority label, with ratios 70 / 15 / 15 and seed **42**. Every image in a group receives the same split.
7. Training reads only `data/splits/train.csv` and `val.csv`. Evaluation of generalization must use `test.csv`. `ml.src.train` exits if any split CSV is missing or empty.

Image-level random shuffles are disallowed. The uneven image counts (6374 / 272 / 270) are the cost of keeping ~14–20 augmentations of one lesion together. Val/test metrics, when they exist, must be interpreted with those sample sizes — especially 52 positive val images and 52 positive test images.

Python, NumPy, and PyTorch seeds are set to 42; `cudnn.deterministic = True`, `cudnn.benchmark = False`.

---

## 7. Preprocessing

Implemented in `ml/src/transforms.py`. Input to every backbone is a 224×224 RGB tensor with ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.

**Train only (mild):**

- Resize to 224
- Horizontal flip, *p* = 0.5
- Rotation ±10°
- Random resized crop, scale 0.85–1.0
- Brightness/contrast jitter 0.1
- `ToTensor` + ImageNet normalize

**Val, test, and API inference:**

- Resize 224, center crop 224, `ToTensor`, ImageNet normalize

**Deliberately unused:** vertical flip, elastic deform, heavy colour jitter, MixUp/CutMix. Aggressive colour or geometry can invent lesion appearances that do not occur clinically and can inflate training curves without helping a real photograph.

API uploads are decoded with Pillow to RGB, then the same val/test transform is applied (`ml/src/predict.py`). Softmax is over two logits. Probability keys: `monkeypox`, `non_monkeypox`.

---

## 8. Transfer Learning

Small dermatology sets overfit if a full ImageNet CNN is trained from epoch 1. Default YAML (`ml/config/*.yaml`, `ml/src/train.py`):

| Setting | Default |
| --- | --- |
| Pretrained | ImageNet (`pretrained: true`) |
| Dropout on the new head | 0.2 |
| Epochs (max) | 30 |
| Freeze epochs (phase 1) | 5 |
| Batch size | 16 |
| Head learning rate | 0.001 |
| Backbone learning rate (phase 2) | 0.0001 |
| Weight decay | 0.0001 |
| Optimizer | AdamW (separate param groups) |
| Loss | Cross-entropy |
| Early stopping | patience 7 on **`val_recall_monkeypox`** |
| Mixed precision | off by default (CUDA AMP only if enabled) |
| Seed | 42 |

**Phase 1.** `freeze_backbone`: all weights frozen except the classification head. The head is trained at `lr_head`.

**Phase 2.** `unfreeze_last_blocks` — **not** a full backbone unfreeze:

- ResNet-50: `layer4`
- DenseNet-121: `denseblock4` (+ `norm5` if present)
- EfficientNet-B0 / MobileNetV3-Small: last two `features` blocks

Those stages train at `lr_backbone`; the head stays at `lr_head`. Best checkpoint is written when validation recall on the positive class improves: `ml/artifacts/checkpoints/{model.name}_best.pt`.

Rationale: fit a linear-ish head on frozen features first, then adapt high-level filters slowly so ImageNet structure is not erased in one step, while keeping most of the encoder stable on a few thousand unique photographs.

---

## 9. Model Architectures

Registry: `ml/src/models.py`. Backbones live under `ml/src/architectures/`. Four candidates:

| Config YAML | Registry name | Backbone | New head | Grad-CAM target |
| --- | --- | --- | --- | --- |
| `ml/config/efficientnet_b0.yaml` | `efficientnet_b0` | EfficientNet-B0 (ImageNet-1K V1) | `Dropout → Linear(2)` replacing `classifier` | last `features` block |
| `ml/config/densenet121.yaml` | `densenet121` | DenseNet-121 | `Dropout → Linear(2)` | `features.denseblock4` |
| `ml/config/resnet50.yaml` | `resnet50` | ResNet-50 | `Dropout → Linear(2)` replacing `fc` | `layer4` |
| `ml/config/mobilenet_v3.yaml` | `mobilenet_v3_small` | MobileNetV3-Small | two-layer head: `Linear → Hardswish → Dropout → Linear(2)` | last `features` block |

All are two-class networks. There is **no** multi-class differential head. MobileNetV3-Small is the CPU-friendly candidate; EfficientNet-B0 is the default name used by the API when nothing is selected.

**No architecture is designated as the production model.** Selection happens only after real test metrics exist.

---

## 10. Evaluation Methodology

Positive class = **monkeypox (1)**. Confusion matrix layout `[[TN, FP], [FN, TP]]`. Default decision threshold **0.5** unless a later protocol documents a different operating point.

**Reported when a real evaluation exists** (`ml/src/metrics.py`, `ml/src/evaluate.py`):

- Accuracy, precision, recall
- **Sensitivity** (same as recall on the positive class)
- **Specificity**
- F1
- **ROC-AUC**, **PR-AUC**
- Confusion matrix
- p50 inference latency (ms)
- Plots: ROC, PR, confusion matrix, train/val loss (`ml/artifacts/plots/`)

Accuracy alone is insufficient: the kept set is imbalanced (~1.66:1) and a missed mpox case (false negative) is treated as costlier than an extra false positive for *model selection*, which is why early stopping and ranking lead with recall. Specificity guards against a trivial “always monkeypox” policy. ROC-AUC and PR-AUC evaluate ranking, not a single 0.5 cut.

Evaluation must load a checkpoint and the **named split CSV**. Evaluating on all of `data/raw` without splits is disallowed.

### 10.1 Model selection policy

`python -m ml.scripts.compare_models` reads `*_test.json` files whose `accuracy` is not null, ranks them, writes `ml/artifacts/metrics/selection.json`, and copies the winner to `selected.pt` when a checkpoint file exists.

Priority (in order):

1. Recall / sensitivity for monkeypox  
2. F1  
3. ROC-AUC, then PR-AUC  
4. Specificity (must not collapse)  
5. Smaller |val − test| gap on F1 (overfit penalty)  
6. Inference latency  

### 10.2 Model results

**PENDING REAL TRAINING RESULTS.**

Current `ml/artifacts/metrics/selection.json`:

```json
{
  "status": "pending_training",
  "selected_model": null,
  "checkpoint": null,
  "reason": "No completed experiment metrics. Add images under data/raw and run training.",
  "ranking": []
}
```

Placeholder files `efficientnet_b0_test.json`, `densenet121_test.json`, `resnet50_test.json`, and `mobilenet_v3_small_test.json` still have `n_samples: 0` and all metric fields `null`. Their older `notes` string (“Pending training — no dataset.”) is leftover: **the dataset and splits are present**; **scores are not**. Nulls are not zeros and not unpublished accuracies.

No `*_best.pt` or `selected.pt` is in `ml/artifacts/checkpoints/` at the time of this report. The API therefore runs in **demo mode** unless `DEMO_MODE=false` (which would return HTTP 503).

Do not copy numbers from Setegn & Dejene, from other mpox CNN papers, or from demo-mode softmax outputs into this section.

---

## 11. Explainable AI / Grad-CAM

Tabular XAI in the BMC 2025 paper (SHAP, LIME, ELI5, counterfactuals, QLattice) attributes a symptom-model score to named clinical features. This project has **no symptom vector**. Explainability is **spatial**.

Implementation: `ml/src/gradcam.py` (Selvaraju et al., Grad-CAM). Forward hook on the architecture’s target conv layer; backward hook on the predicted (or requested) class score; channel weights = spatial mean of gradients; ReLU of the weighted combination; bilinear upsample to the input size; min–max normalize to [0, 1]; jet colormap overlay (alpha 0.45) encoded as PNG base64.

API: `POST /api/v1/predict?explain=true` (default true). Response `explanation.method` is `gradcam`.

**What Grad-CAM means.** Regions that influenced the class logit — a localization hint for teaching.

**What Grad-CAM does not mean.** It does not name a vesicle as pathognomonic, does not prove etiology, does not replace PCR, and is not comparable to SHAP on fever or lymphadenopathy. Known failure modes: highlighting rulers, bandages, text, nails, or background contrast; missing the true lesion; looking locally coherent while the decision is still a dataset artefact.

**Demo mode** draws a **placeholder radial heatmap** from a hash of image bytes and sets `explanation.note` to state that it is not trained Grad-CAM. The UI must show that note.

---

## 12. System Architecture (brief)

Three-tier research stack. Details and diagrams: **[architecture.md](architecture.md)**. Interfaces: [contracts.md](contracts.md).

| Tier | Path | Role |
| --- | --- | --- |
| Client | `client/` | Next.js App Router (TypeScript), port 3000 |
| API | `server/` | FastAPI `app.main:app`, port 8000 |
| ML | `ml/` | Audit, splits, train, evaluate, Grad-CAM, artifacts |

Compose project name: `poxscanner` (`docker-compose.yml`). Service `server` publishes 8000; `client` publishes 3000. The browser calls `http://localhost:8000` (`NEXT_PUBLIC_API_BASE_URL`), not the Compose DNS name `server`, because the URL is inlined into the client bundle.

**Request flow (trained checkpoint, when one exists):** UI multipart upload → FastAPI validation → decode RGB → ImageNet 224 preprocess → CNN softmax → optional Grad-CAM → JSON with display label, probabilities, disclaimer, overlay.

**Request flow (current default):** no checkpoint → deterministic demo stub → `demo_mode: true` + placeholder heatmap.

---

## 13. Backend

FastAPI application in `server/app/`. Version `0.1.0`. CORS from `CORS_ORIGINS` (default `http://localhost:3000`). Lifespan hook attempts checkpoint load via `InferenceService.try_load()`.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | `status`, `service: poxscanner-api`, `model_loaded`, `demo_mode`, `version` |
| `GET /api/v1/model-info` | Name, task `binary_skin_lesion_classification`, classes, 224 input, MIME list, `metrics` (null until real JSON), `selection_status`, `demo_mode`, disclaimer |
| `POST /api/v1/predict` | Multipart field `image`; query `explain` default true |

Checkpoint resolution (`server/app/services/inference.py`): `MODEL_CHECKPOINT` if set; else `ml/artifacts/checkpoints/selected.pt`; else `selection.json` `checkpoint`; else any `*_best.pt`.

`DEMO_MODE=auto|true|false` (default `auto`): stub if no file; always stub; or require a checkpoint (503 `model_unavailable` if missing).

Demo probabilities are a **SHA-256 of upload bytes** mapped to two logits then softmax — deterministic, not clinical.

Errors are JSON `{ "error": { "code", "message" } }`: 400 `invalid_image` / `unsupported_media_type` / `empty_file`; 413 `file_too_large`; 422 validation; 503 only when demo is disabled and no model is loaded.

---

## 14. Frontend

Next.js App Router in `client/`. Shared types in `client/types/api.ts` match the predict contract.

- **Header:** API reachability, model name/version, checkpoint vs no-checkpoint.
- **Disclaimer** (always visible): *Research and educational use only. This tool does not provide a medical diagnosis.*
- **Demo banner** when `demo_mode` is true: *Demo mode — no trained checkpoint loaded. Results are placeholders, not model output.*
- **Uploader:** drag-and-drop or file picker; JPEG/PNG/WebP; max 8 MB (client-side check aligned with the API); local object-URL preview; Analyze / Clear.
- **Results:** display label (`Monkeypox` / `Non-Monkeypox`), confidence as a percentage, probability bars for both classes, original vs Grad-CAM overlay, optional explanation note.

The page does not persist uploads. `NEXT_PUBLIC_API_BASE_URL` (fallback `NEXT_PUBLIC_API_URL`) defaults to `http://localhost:8000`.

---

## 15. Security

This is a local research demo, not a hardened multi-tenant clinic system. Controls that *are* implemented:

- **Upload allowlist:** MIME `image/jpeg`, `image/png`, `image/webp`; Pillow decode; MIME vs actual format must match; empty files rejected; max **8 000 000** bytes; max dimension **4096** px (decompression-bomb / huge-image guard).
- **Filename untrusted:** client filename is ignored; not written to disk by default.
- **No default persistence:** `STORE_UPLOADS=false`. Enabling storage writes UUID-named files under `server/var/uploads/` and is an explicit, auditable choice.
- **No patient identity fields** in request or response schemas.
- **CORS** restricted to configured origins (default localhost UI).
- **Demo vs real inference** is a first-class flag so stub output cannot be silently presented as a checkpoint.
- **Checkpoints mounted read-only** in Compose (`./ml/artifacts/checkpoints:/app/ml/artifacts/checkpoints:ro`).
- **No EHR, accounts, or clinical audit log.** Do not put real patient photographs on a public server without a legal and ethics review.

These measures reduce accidental leakage of uploads; they do not make the classifier a medical device or a secure hospital integration.

---

## 16. Limitations

See [ethics-and-limitations.md](ethics-and-limitations.md) for the full discussion. Summary:

- **PENDING REAL TRAINING RESULTS** — no accuracy, AUC, or F1 to report.
- **Image-only.** No fever, lymphadenopathy, exposure, vaccination, or immune status.
- **Binary head.** Chickenpox, HFMD, measles, healthy skin, and other negatives share one label; the model cannot name the alternative.
- **Small val/test at group level** (272 / 270 images, 52 positives each). Even after training, intervals will be wide.
- **Label noise and 23 cross-class near-duplicate groups** remain.
- **Bias** (skin tone, lesion stage, lighting, geography, clade, web-canonical photos) is **unevaluated**.
- **224×224 crop** discards dermoscopy-scale detail; single image, no time course.
- **No prospective clinical trial.** Future test-split scores would still be retrospective.
- **Grad-CAM** is not a causal or diagnostic explanation.
- **Demo mode** is a hash stub.
- Do not quote the BMC 2025 89.3% LGBM figure as this project’s performance.

---

## 17. Ethical Considerations

Skin photographs can identify people (faces, tattoos, backgrounds, metadata). Default design: do not store uploads; do not log pixel payloads; do not ship EXIF GPS into manifests.

Outputs must not be used for isolation, treatment, vaccination, reporting, or triage. Students and examiners should treat the UI as a **pipeline demonstration**.

If darker phototypes, early lesions, or atypical 2022+ presentations are under-represented — as is common in public rash sets — error rates may be unequal. That is a reason **not** to deploy, not a reason to invent fairness metrics that were never computed.

Responsible communication: lead with sensitivity/specificity and sample sizes when results exist; keep demo and trained inference visually distinct; keep the disclaimer; cite dataset licences if a public set is used.

---

## 18. Future Work

1. Complete training of all four backbones on the existing splits; run `evaluate` on **test**; fill metrics JSON with **real** numbers only.
2. Run `compare_models` and record the selected checkpoint, ranking table, and val–test F1 gap.
3. Report 95% confidence intervals (or bootstrap) for sensitivity/specificity given *n* = 52 test positives.
4. Manually review the 23 cross-class near-duplicate groups; document keep/drop decisions.
5. If labels allow, slice errors by filename family (MKP / HFMD / CHP / …) and, where possible, by skin tone or source.
6. Calibration plots and threshold selection other than 0.5, documented before changing the API.
7. Optional: multi-class head if the negative bucket is split into named dermatoses — a different task, not a silent metric boost.
8. Optional: fuse image scores with a **separate**, consented symptom form (closer to Setegn & Dejene), still non-diagnostic.
9. Do not scrape social media or unidentified clinical sites for more images.

Until step 1–2 complete, every public-facing sentence about “how accurate the model is” must remain: **PENDING REAL TRAINING RESULTS.**

---

## References (selected)

- Setegn GM, Dejene BE. Explainable AI for Symptom-Based Detection of Monkeypox: a machine learning approach. *BMC Infectious Diseases*. 2025;25:419. https://doi.org/10.1186/s12879-025-10738-4
- Selvaraju RR, et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV 2017.
- Mitchell M, et al. Model Cards for Model Reporting. FAT* 2019.
- Ali SN, et al. Monkeypox Skin Lesion Detection Using Deep Learning Models: A Preliminary Feasibility Study. arXiv:2207.03342. (MSLD; cite if those images are used.)

Internal: [dataset.md](dataset.md), [methodology.md](methodology.md), [model-card.md](model-card.md), [ethics-and-limitations.md](ethics-and-limitations.md), [architecture.md](architecture.md), [contracts.md](contracts.md).
