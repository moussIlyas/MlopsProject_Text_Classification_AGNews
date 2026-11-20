"""Explicit upload of local artifacts to the Hugging Face S3 gateway."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def upload_artifacts(cfg: dict[str, Any], artifact_dir: Path, run_id: str) -> list[str]:
    """Upload artifact/ to s3://<bucket>/<prefix>/<run_id>/ and return the keys.

    Credentials are read from the environment (HF_S3_ACCESS_KEY / HF_S3_SECRET_KEY),
    never from config.yaml or from the Docker image.
    """
    hcfg = cfg.get("hf", {})
    if not hcfg.get("upload", False):
        return []
    access_key = os.getenv("HF_S3_ACCESS_KEY")
    secret_key = os.getenv("HF_S3_SECRET_KEY")
    if not access_key or not secret_key:
        print("[WARN] HF_S3_ACCESS_KEY / HF_S3_SECRET_KEY are not set: skipping HF upload")
        return []

    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=hcfg["endpoint_url"],
        region_name=hcfg.get("region", "us-east-1"),
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    prefix = f"{str(hcfg['artifact_prefix']).strip('/')}/{run_id}"
    uploaded: list[str] = []
    for path in sorted(artifact_dir.rglob("*")):
        if path.is_file():
            key = f"{prefix}/{path.relative_to(artifact_dir).as_posix()}"
            client.upload_file(str(path), hcfg["bucket"], key)
            uploaded.append(key)
            print(f"[hf] uploaded s3://{hcfg['bucket']}/{key}")
    return uploaded
