"""Minimal Trainer for a scikit-learn estimator."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline


class Trainer:
    """Just fit / predict; no epochs, no device, no framework machinery."""

    def __init__(self, model: Pipeline, cfg: dict[str, Any]) -> None:
        self.model = model
        self.cfg = cfg

    def fit(self, texts: list[str], labels: np.ndarray) -> "Trainer":
        self.model.fit(texts, labels)
        print(f"[train] fitted {self.cfg['model']['name']} on {len(texts)} documents")
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.predict(texts))
