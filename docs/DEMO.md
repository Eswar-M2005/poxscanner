# College demonstration procedure

Deterministic walkthrough for presenting Pox Scanner. Follow the steps in order. Do **not** invent accuracy, AUC, or other trained-model scores. Do **not** treat demo-mode labels as clinical output.

**This is not a medical diagnostic device.** The product disclaimer (API and UI) is:

> Research and educational use only. This tool does not provide a medical diagnosis.

## Status you must state up front

| Examiner label | What is actually true in this repository |
| --- | --- |
| **WAITING_FOR_MODEL** | No **selected** checkpoint. `selection.json` is `pending_training`. A mid-training `efficientnet_b0_best.pt` may appear under `ml/artifacts/checkpoints/` while Claude trains — that is **not** a finished, compared, or justified production model. `GET /health` reports `model_loaded: false` unless a real `.pt` was actually loaded. Default `DEMO_MODE=auto` serves the **stub** until a checkpoint loads. |
| **PENDING REAL TRAINING RESULTS** | Local `data/raw/` **is populated** and leakage-safe splits **exist** (7518 audited, 602 exact dups removed, 6916 unique, 6374/272/270, 127 groups). Placeholder `*_test.json` still have `n_samples: 0` and null scores. `GET /api/v1/model-info` reports `selection_status: "pending_training"` and `metrics: null`. **Do not invent test metrics.** |

Say both labels in the opening sentence. Then prove them from `/health` and `/api/v1/model-info` (steps 3 and 10). Do not quote Setegn & Dejene’s 89.3% LGBM accuracy as this project’s result — that paper is **symptom-based**, not this image pipeline.

---

## Ports, URLs, and environment names

| Role | Default | Source |
| --- | --- | --- |
| API | `http://localhost:8000` | Uvicorn default; Makefile `api` uses `--port ${PORT:-8000}`; Docker `CMD` uses `--port ${PORT:-8000}`; root `.env.example` lists `PORT=8000` (uvicorn does **not** read `PORT` unless you pass `--port`) |
| UI | `http://localhost:3000` | `cd client && npm run dev` → `next dev` |
| CORS | `http://localhost:3000` | Server env `CORS_ORIGINS` |

### Which API URL the client actually uses

The browser client resolves the API base in `client/lib/api.ts` as:

1. `NEXT_PUBLIC_API_BASE_URL` (first)
2. else `NEXT_PUBLIC_API_URL`
3. else the hardcoded fallback `http://localhost:8000`

Root `.env.example` and `client/.env.example` set **both** to `http://localhost:8000`. **Docker Compose only injects `NEXT_PUBLIC_API_BASE_URL`** (build arg + runtime env). README documents `NEXT_PUBLIC_API_BASE_URL`.

For a live demo, set **`NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`**. `NEXT_PUBLIC_API_URL` is a fallback only. Next.js inlines `NEXT_PUBLIC_*` at **build** time for production/Docker images; `npm run dev` reads them from `client/.env` / `client/.env.local` / the process environment.

`NEXT_PUBLIC_USE_MOCK_API` appears in `client/.env.example` but is **not read by any client code**. Do not claim a mock-API switch.

Server-only env (from `server/.env.example` / `server/app/core/config.py`): `APP_ENV`, `APP_VERSION`, `CORS_ORIGINS`, `MAX_UPLOAD_BYTES` (default `8000000`), `STORE_UPLOADS` (default `false`), `DEMO_MODE` (default `auto`), `MODEL_CHECKPOINT`, `LOG_LEVEL`.

---

## Prerequisites (once per machine)

- Python **3.11+**, Node.js **20+**
- From the repository root:

```bash
# optional: copy env files if you need to override defaults
cp .env.example .env
cp server/.env.example server/.env
cp client/.env.example client/.env

python -m pip install -r server/requirements.txt
cd client && npm install && cd ..
```

Leave `DEMO_MODE=auto` and `MODEL_CHECKPOINT=` empty for the **fallback** demonstration (no checkpoint). Leave `STORE_UPLOADS=false` so uploads are not written to disk.

Training images live under `data/raw/` on this machine (gitignored). Do **not** regenerate splits or re-audit during the demo. For the upload demo, use any **JPEG, PNG, or WebP** you are allowed to show, including a file from `data/raw/` or a tiny synthetic PNG kept **outside** the training folders (step 4).

---

## 1. Start the backend

Open **terminal A**. From the repository root, use the README command:

```bash
cd server && uvicorn app.main:app --reload
```

Equivalent Makefile target (binds `0.0.0.0` and honors `PORT`):

```bash
make api
# expands to:
# cd server && PYTHONPATH=..:. uvicorn app.main:app --reload --host 0.0.0.0 --port ${PORT:-8000}
```

`app.main:app` already inserts the repo root on `sys.path`. Default listen address for plain `uvicorn` is `127.0.0.1:8000`. Leave this process running.

**Talking point:** FastAPI app is `server/app/main.py`. Lifespan calls `try_load()` on the inference service. With no checkpoint and `DEMO_MODE=auto`, load is skipped and the API stays in stub mode.

---

## 2. Start the frontend

Open **terminal B**. From the repository root:

```bash
cd client && npm install && npm run dev
```

`package.json` script `dev` is `next dev` (port **3000** unless Next.js reports otherwise). Open `http://localhost:3000`.

You should see:

- Title **Pox Scanner**
- Header badge **API reachable** and **no checkpoint** (because `model_loaded` is false)
- Always-visible disclaimer
- Demo banner: *Demo mode — no trained checkpoint loaded. Results are placeholders, not model output.*

If the badge says **API unreachable**, the UI cannot reach `NEXT_PUBLIC_API_BASE_URL` (fix terminal A / URL / CORS before continuing).

**Talking point:** The client does not run the model. It `fetch`es `GET ${API_BASE}/health`, `GET ${API_BASE}/api/v1/model-info`, and `POST ${API_BASE}/api/v1/predict?explain=true` with FormData field `image`.

---

## 3. Health check (`GET /health`)

In **terminal C**:

```bash
curl -sS http://localhost:8000/health
```

With no checkpoint and `DEMO_MODE=auto` (the college default), the JSON matches the contract:

```json
{
  "status": "ok",
  "service": "poxscanner-api",
  "model_loaded": false,
  "demo_mode": true,
  "version": "0.1.0"
}
```

Point at the fields:

- `status: "ok"` — process is up
- `model_loaded: false` — **WAITING_FOR_MODEL**
- `demo_mode: true` — stub path is active; this is **not** a trained CNN

Optional confirmation of **PENDING REAL TRAINING RESULTS**:

```bash
curl -sS http://localhost:8000/api/v1/model-info
```

Expect `selection_status: "pending_training"`, `metrics: null`, `demo_mode: true`, `task: "binary_skin_lesion_classification"`, `classes: ["non_monkeypox", "monkeypox"]`, `input.image_size: 224`, formats `image/jpeg`, `image/png`, `image/webp`, and the research disclaimer string.

---

## 4. Upload an image

In the UI (`http://localhost:3000`):

1. Drop or browse a **JPEG, PNG, or WebP** file, size **≤ 8 MB** (`MAX_UPLOAD_BYTES=8000000`).
2. Confirm the thumbnail, filename, and size appear.
3. Do **not** click Analyze yet — show examiners that the file is held in the browser until they request inference.

Allowed types are the MIME allowlist (`image/jpeg`, `image/png`, `image/webp`). BMP is allowed in **training data layout** docs, not on this upload API.

If you only need a valid PNG for a curl demo (not a clinical photo), generate one **outside** `data/raw/`:

```bash
python - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, ".")
from tests.fixtures.make_image import write_tiny_rgb_png
write_tiny_rgb_png(Path("/tmp/poxscanner-demo.png"))
print("/tmp/poxscanner-demo.png")
PY
```

**Talking point:** Default `STORE_UPLOADS=false` — the API ignores the client filename and does not persist the bytes. This is a privacy design choice for skin photographs, not a claim of HIPAA certification.

---

## 5. Analyze

Click **Analyze** in the UI (or curl, same contract).

The page shows *Running inference…* then a **Results** panel: predicted class, confidence, two probability bars, model name/version, original image, overlay.

Equivalent API call (field name **must** be `image`; `explain` defaults to true):

```bash
curl -sS -F "image=@/tmp/poxscanner-demo.png;type=image/png" \
  "http://localhost:8000/api/v1/predict?explain=true"
```

Do **not** read a class name aloud as a diagnosis. Read the **flags**:

- `demo_mode` must be `true` in this fallback demo
- `disclaimer` must be the research sentence
- `prediction` is one of `Monkeypox` | `Non-Monkeypox` (display labels)
- `probabilities.monkeypox` + `probabilities.non_monkeypox` ≈ 1
- `confidence` is the probability of the **predicted** class, in `[0, 1]`

The **same file bytes** produce the **same** probabilities (hash-based stub). Re-upload the same file to show determinism. A different file may get a different stub label — that is expected and is **not** evidence the network “saw” a lesion.

**Do not fabricate** a specific percentage in this document or in speech before the JSON appears on screen. Report only the numbers the running API returns.

---

## 6. Explain confidence

On the Results panel, confidence is formatted as a one-decimal percentage of that `[0, 1]` value (e.g. `0.73` → `73.0%`).

Script:

1. Confidence is **not** “probability of disease in the population.” It is the softmax mass on the **argmax class** for this single image.
2. The two bars are the two class probabilities; they are a partition of 1, not two independent tests.
3. In **demo mode**, those numbers come from **SHA-256 of the file bytes** → two logits in \([-4, 4]\) → softmax (`server/app/services/inference.py`, `demo_probabilities`). They are **not** CNN outputs.
4. A high stub confidence is still **not** a trained-model result. **PENDING REAL TRAINING RESULTS** means we have no validated sensitivity/specificity to compare this number against.
5. Decision threshold planned for a future trained model is 0.5 unless evaluation later changes it — that is a protocol note, not a measured operating point.

If a real checkpoint were loaded (`demo_mode: false`, `model_loaded: true`), confidence would be the CNN’s predicted-class probability after ImageNet-normalized 224×224 inference. **That path is WAITING_FOR_MODEL today.**

---

## 7. Show Grad-CAM

The UI caption is **Grad-CAM overlay (`gradcam`)**. In demo mode, also read the note under the figures:

> Placeholder heatmap (demo mode). Not a trained Grad-CAM explanation.

That note is required. The overlay is a **radial placeholder** derived from the same file hash, blended onto the RGB image. It is **not** Selvaraju Grad-CAM on a conv layer.

When a checkpoint exists, the server would:

1. Run the CNN
2. Take the target layer from `get_gradcam_target_layer` (EfficientNet-B0 / MobileNetV3 last `features` block; ResNet-50 `layer4`; DenseNet-121 `features.denseblock4`)
3. Backprop the predicted class score, overlay the CAM, return PNG base64

Even on the real path, say: a hotspot means “this region influenced the class logit,” **not** “this vesicle is pathognomonic,” **not** a substitute for SHAP/LIME on symptoms, and **not** proof of etiology. Known failure modes (bandages, rulers, text, background) are documented in `docs/ethics-and-limitations.md`.

`POST /api/v1/predict?explain=false` omits `explanation` (null). The UI always requests `explain=true`.

---

## 8. Explain limitations

Read these aloud; do not skip them to “save time.”

- **Not diagnostic.** PCR and clinical judgement remain the reference. Softmax on a photo is not a case definition.
- **Image-only.** No fever, lymphadenopathy, exposure, vaccination, or immune status. Setegn & Dejene (BMC Infectious Diseases 2025;25:419) warn that image-based detectors may fail in real life and often lacked explainability. This repo is image-based **because the intended dataset is photographs**, and it inherits those limits.
- **Binary bucket.** `non_monkeypox` does not name varicella, HSV, syphilis, or other look-alikes.
- **No metrics yet.** Dataset and splits exist; completed evaluation JSON does not. Null scores are **PENDING REAL TRAINING RESULTS**, not zeros.
- **Grad-CAM ≠ causal XAI.** Localization overlay, not feature importance over named symptoms.
- **Demo stub ≠ model.** If `demo_mode: true`, probabilities are a hash of bytes.
- **Privacy.** Default no upload storage; lesion photos can still identify people (faces, tattoos, EXIF). Do not demo with identifiable patient photos unless you have consent and a licence.
- **No PHI pipeline.** No accounts, no EHR, no patient IDs in the predict schema.

---

## 9. Demonstrate invalid upload handling

Show **client** then **server**. Do not only show the happy path.

### Client (UI)

The uploader (`client/lib/format.ts`, `validateImageFile`) rejects before `POST`:

| Action | Expected UI message |
| --- | --- |
| Empty file | `The selected file is empty.` |
| Larger than 8 MB | `Image is too large. Maximum size is 8 MB.` |
| Wrong type (e.g. `.txt`, `.pdf`, or a MIME outside JPEG/PNG/WebP) | `Please choose a JPEG, PNG, or WebP image.` |

The Analyze button never appears for a rejected file.

### Server (curl) — contract errors `{ "error": { "code", "message" } }`

Wrong MIME (valid PNG bytes labeled as text):

```bash
curl -sS -F "image=@/tmp/poxscanner-demo.png;type=text/plain" \
  http://localhost:8000/api/v1/predict
```

Expect **400** `unsupported_media_type`.

Corrupt bytes labeled as PNG:

```bash
printf 'this is not an image' > /tmp/poxscanner-bad.png
curl -sS -F "image=@/tmp/poxscanner-bad.png;type=image/png" \
  http://localhost:8000/api/v1/predict
```

Expect **400** `invalid_image` — `File could not be decoded as an image.`

Empty body file:

```bash
: > /tmp/poxscanner-empty.png
curl -sS -F "image=@/tmp/poxscanner-empty.png;type=image/png" \
  http://localhost:8000/api/v1/predict
```

Expect **400** `empty_file`.

Oversized payload is **413** `file_too_large` when `len(bytes) > MAX_UPLOAD_BYTES` (default 8 000 000). Missing `image` field is **422** `validation_error`.

**Talking point:** Validation uses declared MIME **and** Pillow decode (format must match MIME; max dimension 4096 px). Extension/filename is not trusted and is not stored.

### Optional: 503 when a model is required

Only if asked “what if demo is off?”:

```bash
# in a separate terminal, after stopping the auto-demo server
cd server && DEMO_MODE=false uvicorn app.main:app --reload
curl -sS -F "image=@/tmp/poxscanner-demo.png;type=image/png" \
  http://localhost:8000/api/v1/predict
```

Expect **503** `model_unavailable` — *No model checkpoint is available and demo mode is disabled.* That is the honest **WAITING_FOR_MODEL** failure, not a fake prediction. Return to `DEMO_MODE=auto` for the rest of the demo.

---

## 10. Explain architecture

Three tiers:

```
Browser (Next.js, :3000)
    GET  /health
    GET  /api/v1/model-info
    POST /api/v1/predict  (multipart field "image")
        →
FastAPI (uvicorn app.main:app, :8000)
    validate MIME/size/decode
    InferenceService: checkpoint if present, else hash stub
        →
ml/  (training + Grad-CAM; not invoked for stub inference)
    data/raw → splits → train/evaluate → artifacts/checkpoints + metrics
```

| Path | Role |
| --- | --- |
| `client/` | App Router UI: upload, disclaimer, demo banner, results + overlay |
| `server/` | FastAPI: health, model-info, predict, CORS, structured errors |
| `ml/src/` | Audit, splits, transforms, four ImageNet backbones, train, evaluate, Grad-CAM |
| `ml/artifacts/` | Checkpoints and metrics JSON (selection still `pending_training`) |
| `data/raw/` | Local class folders (gitignored). This machine is populated; a fresh clone is empty. |

Checkpoint resolution if one existed: `MODEL_CHECKPOINT`, else `ml/artifacts/checkpoints/selected.pt`, else `selection.json` `checkpoint`, else first `*_best.pt`. None of these files are selected today.

Docker alternative (not required if uvicorn + Next already run):

```bash
docker compose up --build
```

Compose project name `poxscanner`. Service `server` publishes **8000**; `client` publishes **3000**. Browser must call `http://localhost:8000` because `NEXT_PUBLIC_API_BASE_URL` is inlined for the **host browser**, not the Compose hostname `server`.

---

## FALLBACK demonstration (real model cannot load)

This is the **expected college path**. There is no selected checkpoint; do not apologize for it — it is the documented state.

1. Keep `DEMO_MODE=auto` (or `true`). Do not set `DEMO_MODE=false` unless you are showing the 503.
2. Confirm `/health`: `model_loaded: false`, `demo_mode: true`.
3. Run one predict. Show `demo_mode: true` in JSON **and** the rust demo banner in the UI.
4. Say this sentence **verbatim** (or equivalent, without softening):

   > This is the **demo_mode stub**. It is **not the trained model**. Probabilities are a **deterministic hash of the image bytes**. The overlay is a **placeholder**, not Grad-CAM. We are **WAITING_FOR_MODEL**. Metrics are **PENDING REAL TRAINING RESULTS**.

5. Prove determinism: predict the same file twice; probabilities match.
6. If examiners ask for “the real number,” answer: **PENDING REAL TRAINING RESULTS.** Splits exist on this machine; completed evaluation JSON does not. After Claude finishes, `python -m ml.src.evaluate` and `python -m ml.scripts.compare_models` must produce real `*_test.json` and a non-pending `selection.json` before any score is quoted.

`DEMO_MODE=true` forces the stub even if a checkpoint file appears. `DEMO_MODE=auto` uses the stub only when load fails or no file is found. Failed checkpoint load is logged and also falls through to the stub when mode is `auto`.

---

## Examiner talking points (short answers)

**What is the task?** Binary classification of one RGB lesion photo: `monkeypox` vs `non_monkeypox` (UI: Monkeypox / Non-Monkeypox). Input 224×224, ImageNet normalization when a CNN is loaded.

**Why not reproduce the BMC 2025 paper?** That work is tabular symptoms + SHAP/LIME/etc. This repo’s intended data are images. We treat their paper as a **warning** about image-only systems, not a scoreboard.

**Which backbones are planned?** EfficientNet-B0 (default display name in demo), DenseNet-121, ResNet-50, MobileNetV3-Small. None are trained here.

**How do you prevent leakage (when data exist)?** SHA-256 exact dedup, pHash near-duplicate groups stay in one split, patient/`group_id` split when known, else stratified 70/15/15. Not demonstrable until `data/raw` has images.

**Is Grad-CAM explainable AI like SHAP?** No. Spatial localization of the class score, not attribution to named clinical features.

**Can we trust the overlay today?** No. Demo note states it is a placeholder.

**Will you store our test photo?** Not with default `STORE_UPLOADS=false`.

**What would “success” look like after training?** `model_loaded: true`, `demo_mode: false`, `selection_status` no longer `pending_training`, `metrics` filled from evaluation JSON, Grad-CAM `explanation.note` null (or a real-CAM failure note). Until then, **WAITING_FOR_MODEL** / **PENDING REAL TRAINING RESULTS**.

---

## What you must not say

- Any invented accuracy, AUC, sensitivity, or “it works 90% of the time”
- “The model diagnosed monkeypox” while `demo_mode` is true
- “The heatmap proves the lesion is mpox”
- That BMP or PDF uploads are accepted by the API
- That the client uses only `NEXT_PUBLIC_API_URL` — it **prefers** `NEXT_PUBLIC_API_BASE_URL`
- That `NEXT_PUBLIC_USE_MOCK_API` switches the UI to mocks (unused)
- That Docker Compose should set the API URL to `http://server:8000` for a **browser-on-the-host** demo
)
