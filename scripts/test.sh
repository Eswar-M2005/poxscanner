#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

paths=()
if [ -d tests ]; then
  paths+=(tests)
fi
if [ -d server/tests ]; then
  paths+=(server/tests)
fi

if [ "${#paths[@]}" -gt 0 ]; then
  python -m pytest "${paths[@]}" -q
else
  echo "No pytest directories yet (tests/ or server/tests/). Skipping pytest."
fi

if [ -f client/package.json ]; then
  if grep -Eq '"lint"[[:space:]]*:' client/package.json; then
    (cd client && npm run lint)
  else
    echo "client/package.json has no lint script. Skipping client lint."
  fi
else
  echo "No client/package.json. Skipping client lint."
fi
