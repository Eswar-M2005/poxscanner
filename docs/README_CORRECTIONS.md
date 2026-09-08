# README corrections (proposed — do not apply to README.md)

Auditor: AGENT 10 (README / runbook). Date: 2026-08-28.

This file is the **only** write target for this pass. Leave `README.md` unchanged so other processes (including an in-flight training job) can keep using it.

Do not invent metrics. Do not train. Do not `pip install` into the env that is currently training.

Source of truth for interfaces: [contracts.md](contracts.md). Local dataset numbers: [dataset.md](dataset.md).

---

## How to read this file

- **P0** — a new machine following `README.md` as written will fail or silently do the wrong thing.
- **P1** — stale or contradictory; a careful reader can work around it.
- **P2** — polish / sibling-doc drift.

Patch hunks below are **proposals only**.

---

## New-machine can-run matrix

Assumption: fresh clone, no images, no venv, no checkpoints. Commands from repo root.

| Step | README says | What actually happens | Verdict |
| --- | --- | --- | --- |
| Python env | “Python 3.11+” | `pyproject.toml` is `requires-python = ">=3.11"`. Dockerfiles pin **3.11-slim**. This Mac’s default `python3` is **3.13.3**. The combo that actually ran train here is **CPython 3.12.5 + torch 2.2.2 + torchvision 0.17.2** (MPS). 3.13 is not a known-good wheel target for that torch pin. README never creates a venv. | **P0 — will mis-install** |
| Dependencies (API) | `cd server && uvicorn …` | Missing `make install-api` / `pip install -r server/requirements.txt`. Uvicorn is not a system package. | **P0 — missing install** |
| Dependencies (ML) | `python -m ml.src.audit` then train | Missing venv + install. `make install-ml` installs **CPU** torch from `https://download.pytorch.org/whl/cpu`, then `pip install -e ".[ml]"`. It does **not** read `ml/requirements.txt`. That extra file is unpinned and lists unused `timm`. | **P0 — two install paths, CPU torch on Mac** |
| Dependencies (web) | `cd client && npm install && npm run dev` | Works if Node 20+ is present. `package-lock.json` exists — `npm ci` is the reproducible command. Next reads env from **`client/`**, not the repo-root `.env`. | **P1 — works with caveats** |
| Dataset placement | “Place images under `data/raw/…`” | Correct layout. Images are **gitignored**; class folders ship `.gitkeep` only. This machine **does** have 7518 JPEGs (see snapshot). Acquisition notes: `scripts/download_msld_notes.md` (no download script). | **OK if images are copied** |
| Audit | `python -m ml.src.audit` | Exit **2** on empty `data/raw`. Exit **0** when files exist, even if the report `status` is `"issues"` (duplicates, etc.). Needs Pillow + repo root on `sys.path`. | **OK once deps + data exist** |
| Splits | Buried under “Related”, not in the main command block | Required before train: `python -m ml.scripts.make_splits`. **No Makefile target.** `data/splits/*.csv` is **not** gitignored. If those CSVs ship in git, a clone with empty `data/raw` will pass the “empty CSV” check and then fail when opening missing files. Locally, splits already exist — **do not regenerate unless authorized.** | **P0 — order + committed-split footgun** |
| Training | `python -m ml.src.train --model efficientnet_b0` | Refuses only if split CSVs are missing/empty — **not** if image files are missing. Needs pandas, pyyaml, tqdm, scikit-learn, matplotlib, torchvision. First epoch downloads ImageNet weights. | **OK after splits + ML deps** |
| Evaluation | `python -m ml.src.evaluate --checkpoint … --split test` | Exit 1 if checkpoint missing (clear message). Placeholder `ml/artifacts/metrics/*_test.json` files already exist with `n_samples: 0` / nulls — they are **not** results. | **OK; do not quote placeholders** |
| Backend | `cd server && uvicorn app.main:app --reload` | Makefile uses `PYTHONPATH=..:.` and `--host 0.0.0.0 --port ${PORT:-8000}`. `server/app/main.py` also inserts the repo root onto `sys.path`, so the README form often works, but it skips host/port/env. Copy **repo-root** `.env.example` → `.env`. `server/.env.example` is a subset. Demo mode works **without** a checkpoint. | **P1 — command incomplete** |
| Frontend | `cd client && npm run dev` | Default `http://localhost:3000`. Client resolves API as `NEXT_PUBLIC_API_BASE_URL` **then** `NEXT_PUBLIC_API_URL` **then** `http://localhost:8000` (`client/lib/api.ts`). Root `.env` is **not** loaded by Next. Copy `client/.env.example` → `client/.env.local`. | **P0 — env var name + file location** |
| Tests | not in README Commands | `make test` → `scripts/test.sh` → `pytest tests server/tests` + `npm run lint` in `client/`. Does **not** run `npm test` (vitest) or `npm run typecheck`. Needs `pytest` (`pip install -e ".[dev]"` or API extras). Tests use tiny generated PNGs; they must not need real data or a checkpoint. | **P1 — undocumented / incomplete** |
| Docker | `docker compose up --build` | Compose project `poxscanner`. Services are **`server`** and **`client`**, Dockerfiles **`Dockerfile.server`** / **`Dockerfile.client`**. Sibling **`Dockerfile.api`** / **`Dockerfile.web`** are stale duplicates. Docs still say services `api` / `web`. | **P0 — two Dockerfile pairs** |

**Bottom line:** a new machine **cannot** follow README as written and get a working Python/ML stack. API+UI demo mode can work after installing deps and copying env files, even with empty `data/raw`. Training requires extra steps README omits (venv, 3.12, MPS vs CPU torch, `make_splits`, data copy).

---

## Local workspace snapshot (2026-08-28) — README Status is stale

`README.md` Status table still says empty `data/raw`, splits not generated, no checkpoints, no metrics.

Observed on this machine (not a published result):

| Item | README | This workspace |
| --- | --- | --- |
| `data/raw/monkeypox/` | Empty | **2828** JPEG |
| `data/raw/non_monkeypox/` | Empty | **4690** JPEG |
| `data/splits/` | Not generated | `train.csv` 6374 rows, `val.csv` 272, `test.csv` 270 + `split_meta.json` (`status: ok`, seed 42) |
| Checkpoints | None | `ml/artifacts/checkpoints/efficientnet_b0_best.pt` may exist from an **interrupted / in-progress** EfficientNet-B0 run on MPS. Not a finished selected model. Gitignored. |
| Test metrics JSON | “Not available” | Files exist but are still **null placeholders** (`n_samples: 0`, note `Pending training — no dataset.`). `selection.json` is still `pending_training`. |
| Selected model | `pending_training` | Still `pending_training` — correct until `python -m ml.scripts.compare_models` sees real metrics. |

`docs/dataset.md` already has the audit/split snapshot. README does not. Sibling `docs/model-card.md` and `docs/methodology.md` still claim empty `data/raw` / no test CSV — also stale, out of scope for this file.

**Do not paste training-loss numbers or an incomplete checkpoint into README as metrics.**

---

## Flagged issues (requested)

### 1. `NEXT_PUBLIC_API_BASE_URL` vs `NEXT_PUBLIC_API_URL`

Canonical in running code: **`NEXT_PUBLIC_API_BASE_URL` first**, fallback `NEXT_PUBLIC_API_URL`, fallback `http://localhost:8000`.

| Location | Variable |
| --- | --- |
| `client/lib/api.ts` | `NEXT_PUBLIC_API_BASE_URL` ?? `NEXT_PUBLIC_API_URL` |
| `docker-compose.yml` build arg + env | `NEXT_PUBLIC_API_BASE_URL` |
| `Dockerfile.client` | `NEXT_PUBLIC_API_BASE_URL` |
| `Dockerfile.web` (stale) | `NEXT_PUBLIC_API_URL` |
| `.env.example` (repo root) | **both** set to `http://localhost:8000` |
| `client/.env.example` | both, plus unused `NEXT_PUBLIC_USE_MOCK_API` |
| `README.md` | `NEXT_PUBLIC_API_BASE_URL` |
| `scripts/dev.sh` | still says `NEXT_PUBLIC_API_URL` |
| `docs/architecture.md`, `docs/contracts.md` | `NEXT_PUBLIC_API_URL`; Compose services `api` / `web` |

Next.js only inlines `NEXT_PUBLIC_*` from files under **`client/`** (e.g. `client/.env.local`). Putting the var only in the repo-root `.env` does nothing for `npm run dev`.

Propose: document **one** name (`NEXT_PUBLIC_API_BASE_URL`), keep the other as a deprecated alias, and tell people to copy `client/.env.example`.

### 2. Missing venv instructions

README never mentions `python -m venv`. `.gitignore` ignores `.venv/`, `venv/`, `env/`.

This workspace uses:

- Repo-root `.venv` — CPython **3.12.5**, `include-system-site-packages = false`, **torch 2.2.2**, MPS available.
- `ml/.venv` — also 3.12.5, created **from** `.venv` with `--system-site-packages` (fragile nested venv). Training was launched as `ml/.venv/bin/python -m ml.src.train`.

System `python3` on PATH is **3.13.3**. `make install-api` / `make install-ml` call bare `python`.

Propose a single venv at repo root, interpreter `python3.12`, activate before any `make` / `python -m` command. Do not recreate or pip-install into the env that is currently training.

### 3. Unpinned `ml/requirements.txt`

```
torch
torchvision
timm
pillow
numpy
pandas
scikit-learn
matplotlib
pyyaml
tqdm
```

Problems:

- No versions → a new machine will not reproduce this workspace.
- **`timm` is unused.** Architectures import `torchvision.models` (`ml/src/architectures/*.py`).
- **`make install-ml` ignores this file** and uses `pyproject.toml` `[project.optional-dependencies] ml` (`torch>=2.1`, `torchvision>=0.16`, …) plus a **CPU** wheel index.
- `server/requirements.txt` is separately loosely pinned (`torch>=2.2`, `torchvision>=0.17`) and **also** installs torch (API image / `install-api`).

Known-good set from this machine:

```
Python 3.12.5
torch==2.2.2
torchvision==0.17.2
numpy==1.26.4
scipy==1.13.1          # keep <1.14 so pip does not pull NumPy 2 (breaks torch 2.2.2)
scikit-learn==1.5.2
```

Propose: README should say “create a 3.12 venv; pin torch/vision/numpy as above; then `pip install -e ".[ml]"` (and `.[server]` / `.[dev]` as needed). Do not `pip install -r ml/requirements.txt` until that file is pinned and `timm` is dropped or justified.” Prefer documenting the **CPU index only for Linux Docker**, not for Apple Silicon local train.

### 4. Two Dockerfiles (api/web vs server/client)

| Pair | Used by `docker-compose.yml`? | Env var |
| --- | --- | --- |
| `Dockerfile.server` + `Dockerfile.client` | **Yes** (services `server`, `client`) | `NEXT_PUBLIC_API_BASE_URL` |
| `Dockerfile.api` + `Dockerfile.web` | **No** | web still uses `NEXT_PUBLIC_API_URL` |

`Dockerfile.api` ≈ `Dockerfile.server` (comment-only diff). `Dockerfile.web` ≈ `Dockerfile.client` except the public API env name.

Compose limitation (still true): the browser on the host must call `http://localhost:8000`. Do not bake `http://server:8000` into the client bundle. Internal server-side fetches from inside the client container would need the Compose DNS name `server` (not `api`).

Propose: README Compose section names services **`server`** / **`client`**, files **`Dockerfile.server`** / **`Dockerfile.client`**, and states that `Dockerfile.api` / `Dockerfile.web` are leftovers (do not delete in this sprint unless a later cleanup is authorized).

---

## Other runbook gaps (not in the four flags)

1. **Command order.** README main block jumps `audit` → `train`. Splits are buried under “Related”. Correct order: place images → audit → **make_splits** → train → evaluate → `compare_models`.
2. **Makefile vs README.** README never mentions `make help`, `make install-*`, `make api`, `make web`, `make test`, `make compose-up`. `scripts/dev.sh` is the better local runbook and is not linked from README.
3. **`make` has no `make_splits` target.**
4. **API command missing `PYTHONPATH`.** Prefer the Makefile line.
5. **`install-ml` CPU wheels** conflict with local MPS training. Document `make install-ml` as “CPU / Docker-like”; local Mac: install torch 2.2.2 **without** the cpu index (default macOS wheel), then `pip install -e ".[ml]"`.
6. **`python -m ml.src.train --model mobilenet_v3_small`** looks for `ml/config/mobilenet_v3_small.yaml`, which does not exist (`ml/config/mobilenet_v3.yaml`). Falls back to `base.yaml` (**efficientnet_b0**). Use `--config ml/config/mobilenet_v3.yaml`.
7. **Tests:** README should mention `make test`. Note vitest (`cd client && npm test`) and `npm run typecheck` are extra. Need `[dev]` extras for pytest if you did not `install-api` (API requirements already include pytest).
8. **Demo mode** section is still accurate (`DEMO_MODE=auto`, stub when no usable checkpoint). After a real `*_best.pt` exists, `auto` will load it — an interrupted EfficientNet file may start serving **non-demo** inference without finished metrics. Worth a warning.
9. **Fresh clone vs this laptop.** README should state that **images are not in git**; emptiness is the clone default, not necessarily this working tree.

---

## Proposed README rewrites

### A. Replace “What this project is not” last bullet (P1)

Current line 17 claims no trained-model metrics because the dataset is empty. That conflates “no published evaluation” with “no images on disk.”

```diff
- Not a claim of clinical accuracy. **No trained-model metrics exist yet** (dataset empty; training pending).
+ Not a claim of clinical accuracy. **No completed evaluation metrics are published.** Placeholder JSON under
+ `ml/artifacts/metrics/` (`n_samples: 0`, null scores) is not a result. Do not invent numbers.
```

### B. Replace the Status section (P0)

~~~~markdown
## Status

Two different pictures:

### Fresh clone (what git ships)

| Item | State |
| --- | --- |
| Training images (`data/raw/`) | **Not shipped.** Class folders exist with `.gitkeep` only. Image binaries are gitignored (`*.jpg` etc.). |
| Splits (`data/splits/`) | Generate with `python -m ml.scripts.make_splits` **after** images are in place. If CSVs are already in the tree, they still require the matching files under `data/raw/`. |
| Checkpoints (`*.pt`) | Gitignored. None in a clean clone. |
| Test metrics | Placeholder JSON only (`n_samples: 0`). **Not experimental results.** |
| Selected model | `pending_training` until `python -m ml.scripts.compare_models` sees completed metrics. |
| UI / API | Usable in **demo mode** without a checkpoint. |

On an empty `data/raw`, `python -m ml.src.audit` exits **2**. `python -m ml.scripts.make_splits` exits **1**. Never synthesize training images.

### This working tree (2026-08-28, not committed as metrics)

Local `data/raw` is populated (2828 `monkeypox` + 4690 `non_monkeypox` valid JPEGs; audit `ml/artifacts/metrics/audit_report.json`). Splits exist (`split_meta.json`: 6374 / 272 / 270 after dropping 602 exact duplicates). EfficientNet-B0 training may be in progress on MPS; an incomplete `efficientnet_b0_best.pt` may be present. **Do not report accuracy/AUC until `python -m ml.src.evaluate` is run on a finished checkpoint and those numbers are reviewed.** See [docs/dataset.md](docs/dataset.md).
~~~~

### C. Replace Requirements (P0)

~~~~markdown
## Requirements

- **CPython 3.12** (recommended). `pyproject.toml` allows `>=3.11`; Docker API images use `python:3.11-slim`.
  Known-good local combo: **Python 3.12.5 + torch 2.2.2 + torchvision 0.17.2** (macOS MPS).
  Do not use the system `python3` if it is 3.13+.
- Node.js **20+** (Next.js 15).
- Docker optional (Compose project name `poxscanner`).

Create a venv **before** installing anything:

    python3.12 -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    python -m pip install -U pip
    pip install numpy==1.26.4 scipy==1.13.1
    pip install torch==2.2.2 torchvision==0.17.2
    pip install -e ".[ml,server,dev]"

Then, still from the repo root, with the venv active:

    make install-web    # cd client && npm install  (prefer npm ci when lockfile present)

`make install-ml` currently forces CPU torch wheels — skip it on Apple Silicon.
`ml/requirements.txt` is unpinned and includes unused `timm`; do not treat it as the source of truth until it is pinned.
~~~~

### D. Replace Commands (P0)

~~~~markdown
## Commands

From the repository root, venv activated. Prefer `make help` / `scripts/dev.sh`.

    # 0) Env files
    cp .env.example .env                    # API (DEMO_MODE, CORS, …)
    cp client/.env.example client/.env.local
    # Edit client/.env.local: NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
    # NEXT_PUBLIC_API_URL is a deprecated alias; the client checks BASE_URL first.

    # 1) Inspect raw data (exit 2 if empty)
    python -m ml.src.audit

    # 2) Write data/splits/{train,val,test}.csv + split_meta.json
    python -m ml.scripts.make_splits

    # 3) Train one backbone (refuses if split CSVs missing/empty — images must exist)
    python -m ml.src.train --model efficientnet_b0
    # equivalent: python -m ml.src.train --config ml/config/efficientnet_b0.yaml

    # 4) Evaluate a finished checkpoint (not the placeholder JSON)
    python -m ml.src.evaluate --checkpoint ml/artifacts/checkpoints/efficientnet_b0_best.pt --split test

    # 5) Rank completed experiments → ml/artifacts/metrics/selection.json
    python -m ml.scripts.compare_models

    # API (Makefile sets PYTHONPATH and port)
    make api
    # equivalent:
    #   cd server && PYTHONPATH=..:. uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    # Health: http://localhost:8000/health

    # UI
    make web
    # equivalent: cd client && npm run dev
    # App: http://localhost:3000  (API base: NEXT_PUBLIC_API_BASE_URL)

    # Tests (pytest + next lint; does not run vitest)
    make test
    # extra: cd client && npm test && npm run typecheck

    # Full stack (services: server :8000, client :3000)
    make compose-up
    # equivalent: docker compose up --build

Related:

    python -m ml.src.predict --image path.jpg --checkpoint ml/artifacts/checkpoints/efficientnet_b0_best.pt
    python -m ml.src.train --config ml/config/densenet121.yaml
    python -m ml.src.train --config ml/config/resnet50.yaml
    python -m ml.src.train --config ml/config/mobilenet_v3.yaml   # not --model mobilenet_v3_small

Default API: `http://localhost:8000`. Default UI: `http://localhost:3000`.

Compose uses **`Dockerfile.server`** and **`Dockerfile.client`**. Ignore leftover `Dockerfile.api` / `Dockerfile.web` unless you are deleting them. The client bundle must use a **browser-reachable** URL (`http://localhost:8000`), not `http://server:8000`.
~~~~

### E. Demo mode — add checkpoint warning (P1)

After the existing `DEMO_MODE=auto` paragraph, add:

~~~~markdown
If any `ml/artifacts/checkpoints/*_best.pt` is present, `DEMO_MODE=auto` will load it.
An interrupted training run can therefore disable stub inference even when test
metrics are still placeholders. Use `DEMO_MODE=true` to force the stub, or
`DEMO_MODE=false` to require a checkpoint (HTTP 503 if missing).
~~~~

### F. Env vars — document both names, one canonical (P0)

Add under Demo mode or Commands:

~~~~markdown
## Environment

Repo-root `.env` is read by FastAPI (`server/app/core/config.py` also checks `server/.env`).
The Next.js app does **not** read the repo-root `.env`. Use `client/.env.local`.

| Variable | Who | Canonical? |
| --- | --- | --- |
| `NEXT_PUBLIC_API_BASE_URL` | Next client (`client/lib/api.ts`), Compose, `Dockerfile.client` | **Yes** |
| `NEXT_PUBLIC_API_URL` | Fallback in `api.ts`; stale `Dockerfile.web` / older docs | Deprecated alias |
| `DEMO_MODE` | API | `auto` / `true` / `false` |
| `MODEL_CHECKPOINT` | API | empty → `selected.pt` then `selection.json` then first `*_best.pt` |
| `CORS_ORIGINS` | API | default `http://localhost:3000` |
| `STORE_UPLOADS` | API | default `false` |

`client/.env.example` also lists `NEXT_PUBLIC_USE_MOCK_API`; nothing in `client/` reads it.
~~~~

---

## Suggested one-line README pointer

At the top of Commands, README could say: *Longer local runbook: `scripts/dev.sh` (prints this help; it is not an installer).*

`scripts/dev.sh` itself should be updated in a later pass: it still says `NEXT_PUBLIC_API_URL` and Compose service `api`.

---

## What not to put in README

- Any accuracy, AUC, F1, or confusion-matrix numbers (none completed).
- Claims that `data/raw` is empty **on this laptop** (it is not) or that it is populated **in git** (images are gitignored).
- `pip install -r ml/requirements.txt` as the primary ML install until that file is pinned and aligned with `pyproject.toml`.
- Compose service names `api` / `web` (they are `server` / `client`).
- `--model mobilenet_v3_small` without `--config ml/config/mobilenet_v3.yaml`.
- Instructions that would `pip install` into, or recreate, the venv currently used for training.

---

## Sibling docs

Already updated this sprint (do not revert to “dataset empty”):

| File | State |
| --- | --- |
| `docs/architecture.md` | Local `data/raw` populated; splits exist; mid-run checkpoint may exist; not MODEL_GATE_READY |
| `docs/methodology.md` | Dataset/splits present; **PENDING REAL TRAINING RESULTS** |
| `docs/model-card.md` | Same |
| `docs/ethics-and-limitations.md` | Same |
| `docs/DEMO.md` | Demo + fallback; dataset-empty claims corrected |

Still stale (leave until README rewrite is authorized):

| File | Drift |
| --- | --- |
| `docs/contracts.md` | `.env.example` snippet omits `NEXT_PUBLIC_API_BASE_URL`; Compose services still named `api` / `web` |
| `scripts/dev.sh` | `NEXT_PUBLIC_API_URL`; implies Compose hostname `api` |
