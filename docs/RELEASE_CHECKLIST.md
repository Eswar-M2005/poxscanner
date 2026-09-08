# Release quality gates

Checkboxes reflect **this workspace at sprint time**. Do not tick ML result gates until Claude finishes and artifacts are verified. Do not invent metrics.

Legend: **DONE** / **PENDING** / **WAITING_FOR_CLAUDE**.

## DATA

- [x] **DONE** Audit real — `ml/artifacts/metrics/audit_report.txt` (7518 files, 0 corrupt, status `issues` due to 23 cross-class near-dup groups).
- [x] **DONE** Dedup verified — 602 exact SHA-256 duplicate groups removed; `7518 − 602 = 6916`.
- [x] **DONE** Leakage zero (CSV columns) — 0 cross-split SHA overlap; 0 `group_id` overlap; 127 groups.
- [x] **DONE** Splits valid — train 6374 / val 272 / test 270. Group-level 70/15/15; val/test mpox n=52 (wide CIs later).

Do **not** regenerate splits.

## ML

- [ ] **WAITING_FOR_CLAUDE** Smoke training — `efficientnet_b0` job running; not a completed four-model sweep.
- [ ] **WAITING_FOR_CLAUDE** Four models trained — only EfficientNet-B0 in flight.
- [ ] **WAITING_FOR_CLAUDE** Metrics real — `*_test.json` still `n_samples: 0` / null fields.
- [ ] **WAITING_FOR_CLAUDE** Checkpoint exists — a mid-run `efficientnet_b0_best.pt` may appear; not selected.
- [ ] **WAITING_FOR_CLAUDE** Manifest exists — `ml/artifacts/model_manifest.json` absent.
- [ ] **WAITING_FOR_CLAUDE** Selected model justified — `selection.json` is `pending_training`.
- [ ] **WAITING_FOR_CLAUDE** Inference works (real weights) — API is demo/stub until load + authorization.
- [ ] **WAITING_FOR_CLAUDE** Grad-CAM works (real CAM) — demo uses a placeholder heatmap and `explanation.note`.

## BACKEND

- [x] **DONE** Tests present (`server/tests/`: health, model-info, demo predict, validation, 503 when demo disabled).
- [ ] **PENDING** Full pytest on a machine with FastAPI installed — root `.venv` used for training does **not** include FastAPI/uvicorn.
- [ ] **WAITING_FOR_CLAUDE** Real model loads.
- [ ] **WAITING_FOR_CLAUDE** Prediction works with real weights.
- [x] **DONE** Validation works in code/tests (MIME, size, Pillow decode, MIME/format match, 8 MB).

## FRONTEND

- [ ] **PENDING** `npm run typecheck` (not re-run in this sprint to save CPU).
- [ ] **PENDING** `npm run lint`
- [ ] **PENDING** `npm test` (Vitest present: format, results panel, api export smoke).
- [ ] **PENDING** `npm run build`
- [ ] **WAITING_FOR_CLAUDE** Real prediction displayed (`demo_mode: false`, banner off).

## E2E

- [ ] **WAITING_FOR_CLAUDE** Image → model → API → UI
- [x] **DONE** (demo path) Grad-CAM overlay field rendered; demo note shown when stub.
- [x] **DONE** Errors handled in UI (`ApiRequestError`, local MIME/size checks).

## DOCUMENTATION

- [x] **DONE** No fake metrics — college report, methodology, model card, demo script use **PENDING REAL TRAINING RESULTS**.
- [x] **DONE** Actual architecture documented — [architecture.md](architecture.md) with pipeline mermaid and demo-mode path.
- [x] **DONE** Limitations documented — [ethics-and-limitations.md](ethics-and-limitations.md).
- [ ] **PENDING** Setup reproducible — README still stale; proposed text in [README_CORRECTIONS.md](README_CORRECTIONS.md). Apply after Claude finishes.

## MODEL_GATE_READY

**NO**

Required together, then wait for integration authorization:

1. Checkpoint `*.pt` (selected or four `*_best.pt`)
2. `ml/artifacts/model_manifest.json`
3. `selection.json` with `status` not `pending_training`
4. Real test metrics (not null placeholders)
5. Optional: `model_comparison.csv`

Until then: **do not integrate** the mid-training EfficientNet file into the API.

Root `tests/conftest.py` now forces `DEMO_MODE=true` (same as `server/tests`) so `pytest tests/` cannot load a mid-run `*_best.pt` and steal MPS from Claude.

## Cleanup later (authorized only — do not delete this sprint)

From the structure audit. Pipeline still uses `data/raw/`, Compose still uses `Dockerfile.server` / `Dockerfile.client`.

- `datsets/` (~130 MB) + root `Data.csv` — staging duplicates, not consumed by train/API
- `Dockerfile.api` / `Dockerfile.web` — unused twins of server/client images
- unused `timm` in `ml/requirements.txt`; unused `NEXT_PUBLIC_USE_MOCK_API`
- empty `tests/api/`; overlapping predict/model-info tests
- Makefile targets still named `api`/`web` while folders are `server`/`client`

## Known footguns (do not rewrite frozen train.py)

- `--model mobilenet_v3_small` looks for `ml/config/mobilenet_v3_small.yaml`, which does not exist. The file is `ml/config/mobilenet_v3.yaml`. Use `--config ml/config/mobilenet_v3.yaml`.
- `DEMO_MODE=auto` will load any `*_best.pt` on disk. An interrupted EfficientNet file can disable the stub even while metrics are still placeholders. Force the stub with `DEMO_MODE=true` for demos until MODEL_GATE_READY.
- Next.js does not read the repo-root `.env`. Copy `client/.env.example` → `client/.env.local` and set `NEXT_PUBLIC_API_BASE_URL`.

## Public-deploy notes (localhost demo is OK)

Codex server is frozen. For an internet-exposed API, later authorized work: cap upload body size even when `UploadFile.size` is unset; add a structured 500 handler. Not a college localhost blocker. See the security audit (unbounded `await image.read()` when size is absent).
