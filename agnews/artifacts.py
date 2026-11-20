"""Local artifacts: model.joblib, confusion_matrix.png, reports."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


class ArtifactWriter:
    """Writes a clean artifact directory for one run."""

    def __init__(self, cfg: dict[str, Any], directory: Path) -> None:
        self.cfg = cfg
        self.directory = directory

    def write(
        self,
        model: Pipeline,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: dict[str, float],
    ) -> Path:
        # /app/artifact is a mounted volume in Docker: never remove the mount
        # point itself (E busy), only clear its contents.
        if self.directory.exists():
            for child in self.directory.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        else:
            self.directory.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, self.directory / self.cfg["artifacts"]["model_file"])

        matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 6))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{self.cfg['model']['name']} — accuracy {metrics['accuracy']:.4f}")
        plt.tight_layout()
        plt.savefig(self.directory / self.cfg["artifacts"]["confusion_matrix"], dpi=150)
        plt.close()

        (self.directory / "classification_report.txt").write_text(
            classification_report(y_true, y_pred, digits=4), encoding="utf-8"
        )
        (self.directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        print(f"[artifact] {self.directory}: " + ", ".join(sorted(p.name for p in self.directory.iterdir())))
        return self.directory
