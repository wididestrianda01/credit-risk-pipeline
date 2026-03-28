"""
model.py
--------
Model training, stratified cross-validation, threshold calibration,
and persistence helpers.

Supported estimators
--------------------
- LightGBM (primary)
- XGBoost (benchmark)
- Logistic Regression (interpretable baseline)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def train(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> object:
    """Train a LightGBM classifier with stratified k-fold CV."""
    # TODO: implement training loop, return fitted model
    raise NotImplementedError


def save_model(model: object, path: str | Path) -> None:
    """Persist model artifact to disk."""
    # TODO: implement
    raise NotImplementedError


def load_model(path: str | Path) -> object:
    """Load a persisted model artifact."""
    # TODO: implement
    raise NotImplementedError
