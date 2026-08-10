"""Split conformal prediction: turns any point-prediction model (XGBoost, LSTM, TCN,
Transformer — anything with a `.predict(X) -> np.ndarray`) into a calibrated prediction
*interval* with a distribution-free coverage guarantee, e.g. "the true RUL falls inside
[predicted - q, predicted + q] at least 90% of the time" — without assuming anything about
the model's error distribution.

Method (Lei et al. 2018 "split conformal"):
1. Fit the point predictor on a training subset, holding out a calibration subset.
2. Score the calibration subset: residual_i = |y_true_i - y_pred_i|.
3. q = the ceil((n+1)(1-alpha))/n empirical quantile of those residuals — the finite-sample
   correction (not just the plain (1-alpha) quantile) is what gives the marginal coverage
   guarantee for finite n.
4. For any new prediction, the interval [pred - q, pred + q] contains the true value with
   probability >= 1 - alpha, on average over draws from the same distribution as calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConformalIntervals:
    alpha: float
    q: float  # calibrated half-width, in RUL cycles

    @property
    def coverage_target(self) -> float:
        return 1 - self.alpha

    @classmethod
    def calibrate(cls, y_cal_true: np.ndarray, y_cal_pred: np.ndarray, alpha: float = 0.1) -> ConformalIntervals:
        y_cal_true, y_cal_pred = np.asarray(y_cal_true, dtype=float), np.asarray(y_cal_pred, dtype=float)
        residuals = np.abs(y_cal_true - y_cal_pred)
        n = len(residuals)
        level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
        q = float(np.quantile(residuals, level, method="higher"))
        return cls(alpha=alpha, q=q)

    def interval(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y_pred = np.asarray(y_pred, dtype=float)
        lower = np.clip(y_pred - self.q, a_min=0, a_max=None)
        upper = y_pred + self.q
        return lower, upper


def evaluate_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    """Empirical coverage (should be >= the target if calibration was valid) and the mean
    interval width (the practical cost of that guarantee — tighter is more useful).
    """
    y_true = np.asarray(y_true, dtype=float)
    covered = (y_true >= lower) & (y_true <= upper)
    return {
        "empirical_coverage": float(np.mean(covered)),
        "mean_interval_width": float(np.mean(upper - lower)),
    }
