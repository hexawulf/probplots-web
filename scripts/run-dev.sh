#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1
exec ./.venv/bin/uvicorn app.api:app --host 127.0.0.1 --port 5009
