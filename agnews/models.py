"""The single model: TF-IDF + LogisticRegression, fitted by scikit-learn."""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_model(cfg: dict[str, Any]) -> Pipeline:
    """Preprocessing and classifier live in one Pipeline (saved as model.joblib)."""
    mcfg = cfg["model"]
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=int(mcfg["max_features"]),
                    min_df=int(mcfg["min_df"]),
                    ngram_range=tuple(mcfg["ngram_range"]),
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=float(mcfg["C"]),
                    max_iter=int(mcfg["max_iter"]),
                    random_state=int(cfg["seed"]),
                ),
            ),
        ]
    )
