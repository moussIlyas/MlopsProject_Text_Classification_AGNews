"""MLflow wiring: metadata -> Neon, run artifacts -> HF bucket."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from .config import flatten_config


def configure_tracking() -> None:
    from .mlflow_config import configure_mlflow

    configure_mlflow()


def make_run_name(cfg: dict[str, Any]) -> str:
    explicit = cfg["mlflow"].get("run_name")
    if explicit:
        return str(explicit)
    return (
        f"{cfg['model']['name']}-f{cfg['data_fraction']}-"
        f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    )


def setup_experiment(cfg: dict[str, Any]) -> None:
    name = cfg["mlflow"]["experiment_name"]
    root = cfg["mlflow"]["artifact_root"]
    client = MlflowClient()
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        client.create_experiment(name, artifact_location=root)
        print(f"[mlflow] created {name!r} with artifact root {root}")
    elif not str(experiment.artifact_location).startswith("s3://"):
        print(
            f"[WARN] experiment {name!r} stores artifacts at {experiment.artifact_location}. "
            "Change mlflow.experiment_name in config.yaml to get a fresh S3-backed experiment."
        )
    mlflow.set_experiment(name)


def log_run(cfg: dict[str, Any], artifact_dir: Path, metrics: dict[str, float]) -> None:
    mlflow.log_params(flatten_config(cfg))
    mlflow.log_metrics(metrics)
    try:
        mlflow.log_artifacts(str(artifact_dir), artifact_path="artifact")
        print(f"[mlflow] artifacts -> {cfg['mlflow']['artifact_root']}")
    except Exception as exc:
        print(f"[WARN] MLflow artifact upload failed: {exc}")
