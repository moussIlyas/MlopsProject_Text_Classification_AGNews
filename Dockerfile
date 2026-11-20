# syntax=docker/dockerfile:1
# Image for the unique training entrypoint: python train.py
# Build:  docker build -t agnews-train .
# Run:    docker run --env-file .env -v "$PWD/artifact:/app/artifact" agnews-train
# The entrypoint runs `dvc pull` at RUNTIME (needs data.dvc + HF_S3_* env vars).
# No GPU, no torch, no CUDA: the image stays small and builds fast.
# Secrets are injected at RUNTIME (--env-file); they are never copied into the image.

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# git is required by DVC inside the container (data.dvc resolution)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_HTTP_TIMEOUT=600

# Cache wheels across rebuilds; lockfile is the version-pinning parade.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen && chmod +x /app/entrypoint.sh

# entrypoint.sh restores data with `dvc pull`, then starts train.py.
ENTRYPOINT ["/app/entrypoint.sh"]
