#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8010}"

python -m uvicorn app:app --host "$HOST" --port "$PORT"
