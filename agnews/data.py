"""Dataset loading and data tests (simple preprocessing: text concatenation)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RawData:
    train_texts: list[str]
    y_train: np.ndarray
    test_texts: list[str]
    y_test: np.ndarray


def load_raw_data(cfg: dict[str, Any], root: Path, seed: int) -> RawData:
    """Read, validate and (optionally) subsample the two CSV files."""
    dcfg = cfg["data"]
    train_path = root / dcfg["train_csv"]
    test_path = root / dcfg["test_csv"]
    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing. The container runs `dvc pull` first, or locally: "
                "`dvc pull` (repo must contain data.dvc)."
            )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    text_cols = list(dcfg["text_columns"])
    label_col = dcfg["label_column"]
    num_classes = int(dcfg["num_classes"])

    for split, df in (("train", train_df), ("test", test_df)):
        missing = set(text_cols + [label_col]) - set(df.columns)
        if missing:
            raise ValueError(f"{split}: missing columns {sorted(missing)}")
        if df.empty:
            raise ValueError(f"{split}: empty dataset")
        if df[text_cols].isna().any().any():
            raise ValueError(f"{split}: null values in text columns")
        labels = df[label_col].dropna().astype(int)
        if len(labels) != len(df) or not labels.between(1, num_classes).all():
            raise ValueError(f"{split}: labels must be integers in 1..{num_classes}")

    fraction = float(cfg["data_fraction"])
    if fraction < 1.0:
        train_df = train_df.sample(frac=fraction, random_state=seed).reset_index(drop=True)
        test_df = test_df.sample(frac=fraction, random_state=seed).reset_index(drop=True)

    def to_xy(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
        texts = df[text_cols].astype(str).agg(" ".join, axis=1).tolist()
        labels = df[label_col].astype(int).to_numpy() - 1  # 1..4 -> 0..3
        return texts, labels

    train_texts, y_train = to_xy(train_df)
    test_texts, y_test = to_xy(test_df)
    print(f"[data] fraction={fraction} train={len(y_train)} test={len(y_test)} seed={seed}")
    return RawData(train_texts, y_train, test_texts, y_test)
