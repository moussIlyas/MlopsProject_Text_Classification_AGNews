"""Reproducibility helper (no GPU, no framework seeding needed)."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Fix the only two random sources still used (sampling + estimator)."""
    random.seed(seed)
    np.random.seed(seed)
