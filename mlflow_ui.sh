#!/usr/bin/env bash
# Start the MLflow UI backed by Neon Postgres, with artifacts served from HF S3.
# Secrets are read from .env and never printed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

for key in NEON_PGUSER NEON_PGPASSWORD NEON_PGHOST NEON_PGPORT NEON_PGDATABASE NEON_SSLMODE; do
  if [[ -z "${!key:-}" ]]; then
    echo "[mlflow_ui] missing $key in .env" >&2
    exit 1
  fi
done

export MLFLOW_DISABLE_AGENT_HINT=1

# MLflow reads S3 through boto3: point it at the HF gateway.
export AWS_ACCESS_KEY_ID="${HF_S3_ACCESS_KEY:-}"
export AWS_SECRET_ACCESS_KEY="${HF_S3_SECRET_KEY:-}"
export MLFLOW_S3_ENDPOINT_URL="${HF_S3_ENDPOINT_URL:-https://s3.hf.co/moussll}"

# Built from the same helper used by train.py, so the UI points to the exact
# backend that receives the runs. The value is never echoed.
URI="$(uv run python -c 'from agnews.mlflow_config import mlflow_tracking_uri; print(mlflow_tracking_uri())')"
export MLFLOW_TRACKING_URI="$URI"

HOST="${MLFLOW_HOST:-0.0.0.0}"
PORT="${MLFLOW_PORT:-5000}"

# Reflect explicit --host/--port passed through to MLflow in the startup message.
DISPLAY_HOST="$HOST"
DISPLAY_PORT="$PORT"
prev=""
for arg in "$@"; do
  case "$prev" in
    --host) DISPLAY_HOST="$arg" ;;
    --port) DISPLAY_PORT="$arg" ;;
  esac
  [[ "$arg" == --host=* ]] && DISPLAY_HOST="${arg#*=}"
  [[ "$arg" == --port=* ]] && DISPLAY_PORT="${arg#*=}"
  prev="$arg"
done

echo "[mlflow_ui] backend: Neon Postgres (URI hidden)"
echo "[mlflow_ui] experiment: ${EXPERIMENT_NAME:-AG_News_Classification_sklearn}"
echo "[mlflow_ui] starting UI on http://${DISPLAY_HOST}:${DISPLAY_PORT}"

exec uv run mlflow server \
  --backend-store-uri "$URI" \
  --host "$HOST" \
  --port "$PORT" \
  --serve-artifacts \
  "$@"
