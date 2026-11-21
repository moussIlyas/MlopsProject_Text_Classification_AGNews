"""Validate the raw dataset before training.

    python src/validate_data.py
"""

from __future__ import annotations

from agnews.config import DEFAULT_CONFIG, load_config
from agnews.data import load_raw_data


def validate() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    raw = load_raw_data(cfg, DEFAULT_CONFIG.parent.parent, int(cfg["seed"]))
    print(f"[validate] train={len(raw.y_train)} test={len(raw.y_test)} OK")


if __name__ == "__main__":
    validate()
