"""Inference entrypoint of the packaged model image.

    docker run <image> "Wall Street stocks rally"
    echo "text..." | docker run -i <image>

The image contains the frozen model.joblib produced by train.py; no network,
no DVC and no secrets are needed at inference time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
CLASSES = ["World", "Sports", "Business", "Sci/Tech"]  # CSV labels 1..4 -> 0..3


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one AG News text")
    parser.add_argument("text", nargs="*", help="text to classify (or read from stdin)")
    parser.add_argument(
        "--model",
        type=Path,
        default=ARTIFACT_DIR / "model.joblib",
        help="path to the frozen sklearn Pipeline",
    )
    args = parser.parse_args()

    text = " ".join(args.text).strip() if args.text else sys.stdin.read().strip()
    if not text:
        parser.error("provide a text as argument or on stdin")

    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model} (artifact/ must be baked into the image)")

    model = joblib.load(args.model)
    label = int(model.predict([text])[0])
    confidence = float(model.predict_proba([text])[0][label])

    print(
        json.dumps(
            {
                "class": CLASSES[label],
                "label": label,
                "confidence": round(confidence, 4),
                "text": text[:120],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
