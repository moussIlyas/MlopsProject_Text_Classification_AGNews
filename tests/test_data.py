"""Lightweight data validation tests, run with: python -m pytest tests/"""

from __future__ import annotations

import pandas as pd
import pytest

from agnews.config import DEFAULT_CONFIG, load_config


def cfg() -> dict:
    return load_config(DEFAULT_CONFIG)


def _load_csv(kind: str) -> pd.DataFrame:
    c = cfg()
    path = c["data"][f"{kind}_csv"]
    return pd.read_csv(path)


def test_train_csv_exists():
    assert _load_csv("train") is not None


def test_test_csv_exists():
    assert _load_csv("test") is not None


def test_required_columns():
    for kind in ("train", "test"):
        df = _load_csv(kind)
        text_cols = cfg()["data"]["text_columns"]
        label_col = cfg()["data"]["label_column"]
        assert set(text_cols + [label_col]).issubset(df.columns), f"{kind}: missing columns"


def test_no_null_texts():
    for kind in ("train", "test"):
        df = _load_csv(kind)
        text_cols = cfg()["data"]["text_columns"]
        assert not df[text_cols].isna().any().any(), f"{kind}: null texts"


def test_labels_range():
    num_classes = cfg()["data"]["num_classes"]
    label_col = cfg()["data"]["label_column"]
    for kind in ("train", "test"):
        df = _load_csv(kind)
        labels = df[label_col].dropna().astype(int)
        assert len(labels) == len(df)
        assert labels.between(1, num_classes).all(), f"{kind}: labels outside 1..{num_classes}"
