#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cat <<EOF
Pox Scanner — local development

Repo: ${ROOT}

1) Env
     cp .env.example .env

2) API  (terminal A)
     make install-api
     make api
   # equivalent:
   #   cd server && PYTHONPATH=..:. uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   Health: http://localhost:8000/health

3) Web  (terminal B)
     make install-web
     make web
   App: http://localhost:3000

ML (optional, needs images under data/raw — see scripts/download_msld_notes.md)
     make install-ml
     make audit
     python -m ml.scripts.make_splits
     make train
     make evaluate

Docker (browser → http://localhost:3000, API via http://localhost:8000):
     make compose-up

Compose limitation: NEXT_PUBLIC_API_URL is http://localhost:8000 so the
browser on the host can reach the published API port. That hostname is not
the Compose service name (api). Server-side calls from inside the web
container would need http://api:8000 instead.
EOF
