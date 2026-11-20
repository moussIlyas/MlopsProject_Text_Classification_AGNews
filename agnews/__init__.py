"""AG News classification — simple scikit-learn MLOps pipeline.

    config     - read / validate / flatten config.yaml
    utils      - seeding
    data       - CSV loading, data tests, text concatenation
    models     - TF-IDF + LogisticRegression pipeline
    training   - fit / predict
    artifacts  - model.joblib, confusion_matrix.png, reports
    tracking   - MLflow experiment / run logging
    storage    - upload artifacts to the HF S3 gateway

The only executable entrypoint is ``python train.py`` at the project root.
"""

__all__ = ["config", "utils", "data", "models", "training", "artifacts", "tracking", "storage"]
