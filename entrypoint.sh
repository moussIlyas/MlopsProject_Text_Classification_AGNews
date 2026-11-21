#!/usr/bin/env bash
# Container entrypoint: restore the versioned dataset, then run the trainer.
# Credentials come from runtime env (docker run --env-file .env), never from
# the image. HF_S3_* are mapped to the AWS_* names that DVC expects.
set -euo pipefail

export AWS_ACCESS_KEY_ID="${HF_S3_ACCESS_KEY:-}"
export AWS_SECRET_ACCESS_KEY="${HF_S3_SECRET_KEY:-}"
export AWS_DEFAULT_REGION="${HF_S3_REGION:-us-east-1}"
export AWS_REGION="${AWS_DEFAULT_REGION}"
# MLflow's boto3 client needs the HF S3 gateway endpoint, otherwise it calls
# real AWS S3 with HF keys -> InvalidAccessKeyId
export MLFLOW_S3_ENDPOINT_URL="${HF_S3_ENDPOINT_URL:-https://s3.hf.co/moussll}"

DATA_DIR="${DATA_DIR:-data}"
TRAIN_CSV="${DATA_DIR}/train.csv"
TEST_CSV="${DATA_DIR}/test.csv"

if [[ ! -f "${TRAIN_CSV}" || ! -f "${TEST_CSV}" ]]; then
    if [[ ! -f data.dvc ]]; then
        echo "[entrypoint] data.dvc is missing: bootstrap the dataset once with" >&2
        echo "  dvc add data/ && dvc push" >&2
        echo "and commit data.dvc so every image/container can dvc pull it." >&2
        exit 1
    fi
    if [[ -z "${AWS_ACCESS_KEY_ID}" || -z "${AWS_SECRET_ACCESS_KEY}" ]]; then
        echo "[entrypoint] HF_S3_ACCESS_KEY / HF_S3_SECRET_KEY not set: cannot dvc pull" >&2
        exit 1
    fi
    echo "[entrypoint] dataset not present -> dvc pull from s3://MLTP-dvc/dvc-store"
    uv run --frozen dvc pull
else
    echo "[entrypoint] dataset already present, skipping dvc pull"
fi

echo "[entrypoint] starting trainer: uv run train $*"
exec uv run --frozen train "$@"
