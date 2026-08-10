"""Weibull AFT survival model for RUL.

Frames the problem the way reliability engineering actually frames it: at any observed
cycle, "how much longer until failure" is a time-to-event outcome. We sample many
(covariates-at-cycle-t, remaining-life-from-t) landmarks across each train engine's full,
uncapped trajectory (every failure is *observed*, event=1) and fit a Weibull accelerated
failure time model on them. A test engine's last reading is then a landmark right-censored
at zero elapsed time -- "survived at least 0 more cycles, true remaining life unknown" --
which is exactly the AFT model's unconditional expectation, no truncation math required.
"""

from __future__ import annotations

import pandas as pd
from lifelines import WeibullAFTFitter

DEFAULT_LANDMARK_STRIDE = 3  # subsample cycles to cut within-unit correlation + dataset size


def build_landmark_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    stride: int = DEFAULT_LANDMARK_STRIDE,
) -> pd.DataFrame:
    """One row per sampled (unit, cycle) landmark: covariates at that cycle + the engine's
    true (uncapped) remaining life from there, treated as an observed event -- every train
    engine in CMAPSS runs to failure, so there's no censoring here regardless of `rul_cap`
    used elsewhere in the pipeline (that cap is a regression-loss trick; the survival model
    wants genuine time-to-failure).
    """
    df = df.sort_values(["unit_number", "time_cycles"])
    max_cycle = df.groupby("unit_number")["time_cycles"].transform("max")
    landmarks = df[df["time_cycles"] % stride == 0].copy()
    landmarks["duration"] = (max_cycle - df["time_cycles"])[landmarks.index]
    landmarks = landmarks[landmarks["duration"] > 0]  # AFT requires strictly positive durations
    landmarks["event"] = 1
    return landmarks[feature_cols + ["duration", "event"]]


def build_censored_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """One row per unit at its last observed cycle: right-censored at zero elapsed time
    from there ("survived at least this long, true remaining life unknown") -- for the
    CMAPSS test set, or for scoring a live engine that hasn't failed yet.
    """
    last = df.sort_values("time_cycles").groupby("unit_number").tail(1).copy()
    last["duration"] = 1e-3  # AFT requires strictly positive durations; ~0 elapsed since landmark
    last["event"] = 0
    return last.set_index("unit_number")[feature_cols + ["duration", "event"]]


def train_survival_model(landmark_frame: pd.DataFrame, penalizer: float = 5.0) -> WeibullAFTFitter:
    """`penalizer` ridge-regularizes the fit -- CMAPSS sensors are highly collinear, which
    without a substantial penalty produces a non-invertible Hessian, unstable coefficients,
    and (worse) a heavy-tailed fitted distribution for a handful of covariate patterns whose
    predicted *mean* remaining life blows up. 5.0 was picked by sweeping 0.1-15 and comparing
    NASA-score stability on FD001 (a couple orders of magnitude better than the un-penalized
    fit, with only a modest RMSE cost) -- see predict_median_rul for the other half of that
    fix.
    """
    model = WeibullAFTFitter(penalizer=penalizer)
    model.fit(landmark_frame, duration_col="duration", event_col="event")
    return model


def predict_median_rul(model: WeibullAFTFitter, covariates: pd.DataFrame) -> pd.Series:
    """Median remaining life for engines observed right now (elapsed time since the
    landmark ~= 0). Uses the median rather than the AFT model's mean/expectation because a
    Weibull fit's mean is highly sensitive to right-tail behavior -- a few covariate
    patterns with a poorly-constrained tail can blow up the mean while barely moving the
    median, and the median is the far more robust point estimate for a skewed distribution.
    """
    return model.predict_median(covariates)
