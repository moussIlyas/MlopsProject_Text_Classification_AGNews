"""Single entrypoint: data -> train -> evaluate -> artifact -> MLflow -> HF.

    python train.py                       # reads config.yaml (1% of data, logreg)
    python train.py --config other.yaml

All logic lives in the ``agnews`` package; this file only orchestrates it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score

from agnews.artifacts import ArtifactWriter
from agnews.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from agnews.data import load_raw_data
from agnews.models import build_model
from agnews.storage import upload_artifacts
from agnews.tracking import configure_tracking, log_run, make_run_name, setup_experiment
from agnews.training import Trainer
from agnews.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sklearn model and publish its artifacts")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="path to the YAML config")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed)
    print(f"[setup] model={cfg['model']['name']} seed={seed} fraction={cfg['data_fraction']}")

    raw = load_raw_data(cfg, PROJECT_ROOT, seed)
    model = build_model(cfg)
    trainer = Trainer(model, cfg).fit(raw.train_texts, raw.y_train)
    y_pred = trainer.predict(raw.test_texts)
    accuracy = float(accuracy_score(raw.y_test, y_pred))
    print(f"[evaluate] accuracy={accuracy:.4f}")

    configure_tracking()
    setup_experiment(cfg)

    artifact_dir = PROJECT_ROOT / cfg["artifacts"]["dir"]
    with mlflow.start_run(run_name=make_run_name(cfg)) as run:
        output_dir = ArtifactWriter(cfg, artifact_dir).write(
            model=model,
            y_true=raw.y_test,
            y_pred=y_pred,
            metrics={
                "accuracy": accuracy,
                "seed": float(seed),
                "data_fraction": float(cfg["data_fraction"]),
            },
        )
        log_run(cfg, output_dir, {"accuracy": accuracy, "data_fraction": float(cfg["data_fraction"])})
        upload_artifacts(cfg, output_dir, run.info.run_id)
        print(f"[done] run_id={run.info.run_id} artifacts={output_dir} accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
