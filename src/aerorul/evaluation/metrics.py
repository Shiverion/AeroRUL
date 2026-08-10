"""Standard CMAPSS evaluation metrics: RMSE and the NASA PHM08 asymmetric scoring function."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The scoring function from the PHM08 challenge / Saxena et al. 2008.

    Penalizes late predictions (predicting more life than the engine actually has, which
    risks an in-service failure) far more heavily than early predictions (predicting less
    life than it has, which just costs an early maintenance action) — d = predicted - actual:

        s(d) = exp(-d/13) - 1   if d < 0  (early prediction)
        s(d) = exp( d/10) - 1   if d >= 0  (late prediction)

    Total score is the sum over all predictions; lower is better. Unlike RMSE this is not
    symmetric, which matches how a maintenance-decision system should actually be judged.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(scores))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Convenience bundle of both standard metrics plus mean absolute error."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "nasa_score": nasa_score(y_true, y_pred),
    }
