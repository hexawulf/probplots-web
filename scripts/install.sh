#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

/usr/bin/python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install fastapi "uvicorn[standard]" numpy scipy matplotlib
