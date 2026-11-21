"""Configuration loading and validation (single source of truth)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "train.yaml"
SUPPORTED_MODELS = ("logreg",)


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Any) -> None:
    if not isinstance(cfg, dict):
        raise ValueError("config must be a YAML mapping")
    for section in ("seed", "data_fraction", "data", "model", "artifacts", "mlflow"):
        if section not in cfg:
            raise ValueError(f"config is missing required section/key: {section!r}")

    fraction = float(cfg["data_fraction"])
    if not 0 < fraction <= 1:
        raise ValueError(f"data_fraction must be in (0, 1], got {fraction}")

    model_name = str(cfg["model"]["name"]).lower()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"model.name={model_name!r}; choose one of {', '.join(SUPPORTED_MODELS)}")
    if int(cfg["data"]["num_classes"]) < 2:
        raise ValueError("data.num_classes must be at least 2")


def flatten_config(cfg: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Turn the nested YAML into MLflow-compatible string parameters."""
    flat: dict[str, str] = {}
    for key, value in cfg.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_config(value, name))
        elif isinstance(value, (list, tuple)):
            flat[name] = json.dumps(value)
        elif value is None:
            flat[name] = "null"
        else:
            flat[name] = str(value)
    return flat
