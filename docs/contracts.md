# Pox Scanner — Integration Contracts

This file is the source of truth for interfaces. Implement against it. Do not invent incompatible shapes.

## Product

Educational/research decision-support web app for binary image classification:

- `monkeypox` → display label `Monkeypox`
- `non_monkeypox` → display label `Non-Monkeypox`

Not a medical diagnostic product.

Literature baseline: Setegn & Dejene, *Explainable AI for Symptom-Based Detection of Monkeypox: a machine learning approach*, BMC Infectious Diseases 25:419 (2025). https://doi.org/10.1186/s12879-025-10738-4

That paper is **symptom-based ML + XAI** and explicitly notes limitations of image-only systems and the need for explainability. This project is **image-based** because the available dataset is lesion images. We still require explainability (Grad-CAM), leakage control, model comparison, and honest limitations.

## Dataset layout

```
data/raw/monkeypox/         # class 1 images
data/raw/non_monkeypox/     # class 0 images
data/processed/             # optional cached tensors / manifests
data/splits/                # split manifests only
  train.csv
  val.csv
  test.csv
  split_meta.json
```

CSV columns (required):

```
path,label,split,sha256,width,height,group_id
```

- `label`: `monkeypox` | `non_monkeypox`
- `group_id`: patient/group id if known, else the file's sha256 (so identical files never cross splits)
- `path`: repo-relative POSIX path

If `data/raw` is empty, audit/split/train must **exit non-zero with a clear message**. Never synthesize training images.

Allowed image types: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.

## Class encoding

```
LABEL_TO_INDEX = {"non_monkeypox": 0, "monkeypox": 1}
INDEX_TO_LABEL = {0: "non_monkeypox", 1: "monkeypox"}
DISPLAY = {"non_monkeypox": "Non-Monkeypox", "monkeypox": "Monkeypox"}
```

Positive class for recall/sensitivity/PR-AUC: **monkeypox (1)**.

## Seeds and reproducibility

Default seed: `42`. Set Python, NumPy, and PyTorch seeds. `cudnn.deterministic = True`, `cudnn.benchmark = False`.

## Split policy

1. Deduplicate exact SHA-256 matches (keep one).
2. If near-duplicates (perceptual hash Hamming distance ≤ 5) exist, assign the **group** to one split only.
3. If a `group_id` / patient metadata file exists, split by group.
4. Otherwise stratified 70/15/15 by label.
5. Never put the same sha256 or pHash group in two splits.

## Transforms

Train only: resize 224, mild HFlip p=0.5, rotation ±10°, random resized crop scale 0.85–1.0, brightness/contrast jitter 0.1.

Val/test/inference: resize 224, center crop 224, ImageNet mean/std.

Do **not** use vertical flip, elastic deform, or heavy color jitter.

## ML config YAML schema (`ml/config/*.yaml`)

```yaml
model:
  name: efficientnet_b0   # one of: efficientnet_b0, densenet121, resnet50, mobilenet_v3_small
  pretrained: true
  num_classes: 2
  dropout: 0.2
train:
  epochs: 30
  freeze_epochs: 5
  batch_size: 16
  num_workers: 2
  lr_head: 0.001
  lr_backbone: 0.0001
  weight_decay: 0.0001
  early_stopping_patience: 7
  early_stopping_metric: val_recall_monkeypox
  seed: 42
  image_size: 224
  mixed_precision: false
paths:
  data_raw: data/raw
  splits_dir: data/splits
  checkpoint_dir: ml/artifacts/checkpoints
  metrics_dir: ml/artifacts/metrics
  plots_dir: ml/artifacts/plots
```

Checkpoint filename: `ml/artifacts/checkpoints/{model.name}_best.pt`

Metrics JSON: `ml/artifacts/metrics/{model.name}_{split}.json`

## Checkpoint payload

```python
{
  "model_name": str,
  "state_dict": OrderedDict,
  "config": dict,
  "val_metrics": dict,
  "epoch": int,
  "seed": int,
  "class_to_idx": dict,
  "image_size": 224,
  "version": "0.1.0",
}
```

## Metrics JSON schema

```json
{
  "model": "efficientnet_b0",
  "split": "test",
  "version": "0.1.0",
  "n_samples": 0,
  "accuracy": null,
  "precision": null,
  "recall": null,
  "sensitivity": null,
  "specificity": null,
  "f1": null,
  "roc_auc": null,
  "pr_auc": null,
  "latency_ms_p50": null,
  "confusion_matrix": [[null, null], [null, null]],
  "notes": "Pending training — no dataset."
}
```

When metrics are real, fill numbers. Confusion matrix is `[[TN, FP], [FN, TP]]` with positive = monkeypox.

Do **not** invent scores. If untrained, leave nulls and explain in `notes`.

## Model selection

Script: `python -m ml.scripts.compare_models`

Priority (in order):

1. recall / sensitivity for monkeypox
2. F1
3. ROC-AUC, then PR-AUC
4. specificity (must not collapse)
5. |val − test| gap on F1 (smaller is better)
6. inference latency

Writes `ml/artifacts/metrics/selection.json`. If no completed experiments:

```json
{
  "status": "pending_training",
  "selected_model": null,
  "checkpoint": null,
  "reason": "No completed experiment metrics. Add images under data/raw and run training.",
  "ranking": []
}
```

## Shared Python signatures

```python
# ml/src/utils.py
def repo_root() -> Path: ...
def set_seed(seed: int) -> None: ...
def load_yaml(path: Path) -> dict: ...
def sha256_file(path: Path) -> str: ...
def get_logger(name: str): ...

# ml/src/transforms.py
def build_transforms(split: str, image_size: int = 224): ...

# ml/src/dataset.py
class LesionDataset(torch.utils.data.Dataset): ...
def make_splits(...) -> None: ...
def load_split_frame(split: str) -> pd.DataFrame: ...

# ml/src/architectures/{efficientnet,densenet,resnet,mobilenet}.py
def build(num_classes: int, pretrained: bool, dropout: float) -> nn.Module: ...
def get_target_layer(model: nn.Module) -> nn.Module: ...

# ml/src/models.py  (registry only — do not reimplement backbones)
def build_model(name: str, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.2) -> nn.Module: ...
def get_gradcam_target_layer(model: nn.Module, name: str) -> nn.Module: ...
def count_parameters(model: nn.Module) -> int: ...

# ml/src/metrics.py
def binary_metrics(y_true, y_prob, threshold=0.5) -> dict: ...
def save_curves(...) -> None: ...

# ml/src/gradcam.py
class GradCAM:
    def __init__(self, model, target_layer): ...
    def generate(self, input_tensor, class_idx=None) -> np.ndarray: ...

def overlay_cam_on_image(image_rgb: np.ndarray, cam: np.ndarray) -> np.ndarray: ...
def cam_to_png_bytes(overlay_rgb: np.ndarray) -> bytes: ...

# ml/src/predict.py
def load_checkpoint(path: Path, device=None) -> tuple[nn.Module, dict]: ...
def predict_image(model, image, config) -> dict: ...  # probabilities keys: monkeypox, non_monkeypox
```

CLI:

```
python -m ml.src.audit
python -m ml.src.train --model efficientnet_b0
python -m ml.src.train --config ml/config/efficientnet_b0.yaml
python -m ml.src.evaluate --checkpoint ml/artifacts/checkpoints/efficientnet_b0_best.pt --split test
python -m ml.src.predict --image path.jpg --checkpoint ...
python -m ml.scripts.compare_models
python -m ml.scripts.make_splits
```

`ml.src.train` must refuse to run if splits are empty / missing, with a non-zero exit.

## HTTP API

Base: FastAPI app `app.main:app` run from `server/`.

CORS origins from env `CORS_ORIGINS` (comma-separated). Default `http://localhost:3000`.

### GET /health

```json
{
  "status": "ok",
  "service": "poxscanner-api",
  "model_loaded": false,
  "demo_mode": true,
  "version": "0.1.0"
}
```

`model_loaded` is true only if a real checkpoint was loaded.

### GET /api/v1/model-info

```json
{
  "name": "efficientnet_b0",
  "version": "0.1.0",
  "task": "binary_skin_lesion_classification",
  "classes": ["non_monkeypox", "monkeypox"],
  "input": {"image_size": 224, "formats": ["image/jpeg", "image/png", "image/webp"]},
  "metrics": null,
  "selection_status": "pending_training",
  "demo_mode": true,
  "disclaimer": "Research and educational use only. This tool does not provide a medical diagnosis."
}
```

When a checkpoint exists, fill `name` from checkpoint, `metrics` from the matching metrics JSON if present, `demo_mode: false`.

### POST /api/v1/predict

`multipart/form-data` field name: `image`

Optional query/form: `explain=true` (default **true**).

Success 200:

```json
{
  "prediction": "Monkeypox",
  "confidence": 0.0,
  "probabilities": {
    "monkeypox": 0.0,
    "non_monkeypox": 0.0
  },
  "model": "efficientnet_b0",
  "version": "0.1.0",
  "demo_mode": true,
  "disclaimer": "Research and educational use only. This tool does not provide a medical diagnosis.",
  "explanation": {
    "method": "gradcam",
    "overlay_png_base64": "<base64 png>",
    "note": null
  }
}
```

- `prediction` is the **display** label.
- `confidence` is the probability of the predicted class, in `[0, 1]`.
- Probabilities must sum to ~1.
- `explanation.overlay_png_base64` is a PNG overlay of Grad-CAM on the original (demo mode may use a placeholder heatmap and set `note` accordingly).
- Do **not** persist uploads unless `STORE_UPLOADS=true`.

Errors (JSON):

```json
{
  "error": {
    "code": "invalid_image",
    "message": "File could not be decoded as an image."
  }
}
```

Status codes:

- 400 `invalid_image` / `unsupported_media_type` / `empty_file`
- 413 `file_too_large`
- 422 validation
- 503 `model_unavailable` only if demo mode is explicitly disabled **and** no checkpoint exists

Validation:

- MIME allowlist: `image/jpeg`, `image/png`, `image/webp`
- Max bytes: `MAX_UPLOAD_BYTES` default 8_000_000
- Decode with Pillow; reject if decode fails
- Ignore client filename; do not write it to disk
- Verify content is actually an image (not just extension)

## Demo mode

Env: `DEMO_MODE=auto|true|false` (default `auto`).

- `auto`: demo if no checkpoint file at `MODEL_CHECKPOINT`
- `true`: always stub inference (for frontend/dev)
- `false`: require checkpoint; else 503

Demo inference must be **deterministic** for a given image bytes (hash-based probabilities), clearly flagged `demo_mode: true`, and never claimed as a trained medical model.

Default checkpoint path: `../ml/artifacts/checkpoints/selected.pt` or `MODEL_CHECKPOINT`. If `selected.pt` missing, try `selection.json` checkpoint field, else any `*_best.pt`.

## Env vars (`.env.example`)

```
APP_ENV=development
APP_VERSION=0.1.0
CORS_ORIGINS=http://localhost:3000
MAX_UPLOAD_BYTES=8000000
STORE_UPLOADS=false
DEMO_MODE=auto
MODEL_CHECKPOINT=
LOG_LEVEL=info
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Frontend contract

- Next.js App Router + TypeScript
- `NEXT_PUBLIC_API_URL` default `http://localhost:8000`
- POST `${API}/api/v1/predict` with FormData `image`
- Show demo_mode banner when response.demo_mode is true
- Disclaimer always visible: `Research and educational use only. This tool does not provide a medical diagnosis.`

Shared TS type (`client/types/api.ts`) must match the predict JSON above.

## Docker

- `api`: FastAPI on 8000, cwd `/app/server` or PYTHONPATH set so `app.main:app` works
- `web`: Next.js on 3000, `NEXT_PUBLIC_API_URL` pointing at `api` for server and browser-friendly localhost for client (document the limitation)
- Compose project name `poxscanner`

## Tests must not require a trained model or real dataset

Use tiny generated PNG fixtures. Do not add fake experiment metrics JSON with invented scores.
