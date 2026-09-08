# Pox Scanner

Educational and research web application for **binary skin-lesion image classification**: `monkeypox` vs `non_monkeypox`.

This is **not** a medical diagnostic device, not a screening product, and not a substitute for laboratory confirmation or clinical judgement. Outputs are research/educational only.

## What this project is

- A client–server–ML stack that classifies a single lesion photograph into two labels (`Monkeypox` / `Non-Monkeypox`).
- A training pipeline for ImageNet-pretrained CNNs (EfficientNet-B0, DenseNet-121, ResNet-50, MobileNetV3-Small) with leakage-controlled splits, conservative augmentation, and Grad-CAM overlays.
- A FastAPI + Next.js demo that can run **without a trained checkpoint** (deterministic stub inference, clearly flagged).

## What this project is not

- Not a diagnostic, screening, or triage device.
- Not a symptom-based predictor. It does not ingest fever, lymphadenopathy, exposure history, or other clinical features.
- Not a claim of clinical accuracy. **No trained-model metrics exist yet** (dataset empty; training pending).

## Literature baseline

Setegn GM, Dejene BE. Explainable AI for Symptom-Based Detection of Monkeypox: a machine learning approach. *BMC Infectious Diseases*. 2025;25:419. https://doi.org/10.1186/s12879-025-10738-4

That paper is **symptom-based** ensemble ML with XAI (SHAP, LIME, ELI5, counterfactuals, QLattice) on tabular clinical features. They explicitly note that prior **image-based** monkeypox detectors have real-life limitations and often lacked explainability.

This implementation is **image-based binary classification** because the intended dataset is lesion photographs, not symptom tables. We take their warning seriously: image-only models miss systemic context, do not replace PCR, and must expose honest limitations plus some form of explanation. We use Grad-CAM as a localization overlay—not as a causal or clinical explanation—and we do not report performance numbers until a real dataset is trained and evaluated.

## Status

| Item | State |
| --- | --- |
| Training images (`data/raw/`) | **Empty** — class folders contain no images |
| Splits (`data/splits/`) | Not generated |
| Checkpoints | None |
| Test metrics (accuracy, AUC, etc.) | **Not available** — do not invent them |
| Selected model | `pending_training` |
| UI / API | Usable in **demo mode** without a checkpoint |

`python -m ml.src.audit`, split generation, and `python -m ml.src.train` must **exit non-zero** while `data/raw` is empty. Never synthesize training images.

## Requirements

- Python **3.11+**
- Node.js **20+**
- Docker (optional; Compose project name `poxscanner`)

## Required data layout

Place images only under the contract paths (repo-relative POSIX). Allowed types: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.

```
data/raw/monkeypox/         # class 1
data/raw/non_monkeypox/     # class 0
data/processed/             # optional caches / manifests
data/splits/
  train.csv
  val.csv
  test.csv
  split_meta.json
```

CSV columns: `path,label,split,sha256,width,height,group_id`

- `label`: `monkeypox` | `non_monkeypox`
- `group_id`: patient/group id if known, else the file SHA-256 (identical files never cross splits)
- Split policy: exact SHA-256 dedup, near-duplicate pHash groups stay in one split, group/patient split when metadata exists, otherwise **stratified 70 / 15 / 15** by label

Details: [docs/dataset.md](docs/dataset.md).

## Commands

From the repository root unless noted.

```bash
# Python 3.12 is recommended for the backend runtime.

# Inspect raw data (fails clearly if empty)
python -m ml.src.audit

# Train one backbone (refuses if splits missing/empty)
python -m ml.src.train --model efficientnet_b0

# Evaluate a checkpoint
python -m ml.src.evaluate --checkpoint ml/artifacts/checkpoints/efficientnet_b0_best.pt --split test

# API (from server/)
cd server && uvicorn app.main:app --reload --port 8000

# UI (from client/)
cd client && npm install && npm run dev

# Full stack
docker compose up --build
```

Related:

```bash
python -m ml.src.train --config ml/config/efficientnet_b0.yaml
python -m ml.src.predict --image path.jpg --checkpoint ...
python -m ml.scripts.make_splits
python -m ml.scripts.compare_models
```

Default API: `http://localhost:8000`. Default UI: `http://localhost:3000` (`NEXT_PUBLIC_API_BASE_URL`).

## Runtime modes

The API reads the authoritative model artifacts from `ml/artifacts/metrics/model_manifest.json`,
`ml/artifacts/metrics/selection.json`, and `ml/artifacts/metrics/threshold_analysis.json`, then loads
`ml/artifacts/checkpoints/selected.pt` once at startup when available.

With `DEMO_MODE=auto` (default), the API serves a deterministic fallback only when no checkpoint is
available. Responses set `demo_mode: true`. This is for UI/API development only and is not a trained
medical model.

Set `DEMO_MODE=false` to require the checkpoint and return HTTP 503 if it is missing.

Uploads are **not stored** unless `STORE_UPLOADS=true`.

## Documentation

| Document | Contents |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | Client–server–ML layout, API, Grad-CAM flow |
| [docs/dataset.md](docs/dataset.md) | Layout, leakage rules, empty-data behaviour |
| [docs/methodology.md](docs/methodology.md) | Transfer learning, metrics, selection, contrast with BMC 2025 |
| [docs/model-card.md](docs/model-card.md) | Intended use, data, pending metrics |
| [docs/ethics-and-limitations.md](docs/ethics-and-limitations.md) | Non-diagnostic use, bias, privacy, Grad-CAM limits |
| [docs/contracts.md](docs/contracts.md) | Integration contracts (source of truth for interfaces) |

Always-visible product disclaimer: *Research and educational use only. This tool does not provide a medical diagnosis.*
