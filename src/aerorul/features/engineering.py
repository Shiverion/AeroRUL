"""RUL labeling, condition normalization, rolling-window features, and a simple health
indicator for the CMAPSS turbofan dataset.

Pipeline: raw sensor readings -> RUL labels -> per-condition normalization -> rolling
statistics -> health indicator -> feature matrix ready for modeling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from aerorul.data.schema import LOW_VARIANCE_SENSORS_SINGLE_CONDITION, SENSOR_COLS, SETTING_COLS

DEFAULT_RUL_CAP = 125  # standard piecewise-linear degradation cap used across CMAPSS literature
DEFAULT_ROLLING_WINDOWS = (5, 10, 20)


def add_rul_train(df: pd.DataFrame, rul_cap: int | None = DEFAULT_RUL_CAP) -> pd.DataFrame:
    """Label each row of a training trajectory with its remaining useful life.

    RUL = (last cycle of that unit) - (current cycle). Engines degrade negligibly early in
    life and roughly linearly near failure, so RUL is capped at `rul_cap` (piecewise-linear
    degradation model) — this also keeps early-life rows from dominating the regression loss
    with a target that a model has no signal to predict anyway.
    """
    df = df.copy()
    max_cycle = df.groupby("unit_number")["time_cycles"].transform("max")
    rul = max_cycle - df["time_cycles"]
    if rul_cap is not None:
        rul = rul.clip(upper=rul_cap)
    df["RUL"] = rul
    return df


def add_rul_test(
    df: pd.DataFrame, rul_truth: pd.Series, rul_cap: int | None = DEFAULT_RUL_CAP
) -> pd.DataFrame:
    """Label every row of a test trajectory with RUL, using the known truth at the last cycle.

    The test set is truncated before failure, so RUL at the last observed cycle is
    `rul_truth[unit]` (not 0) — earlier cycles get that plus however many cycles remain
    until the truncation point.
    """
    df = df.copy()
    max_cycle = df.groupby("unit_number")["time_cycles"].transform("max")
    unit_truth = df["unit_number"].map(rul_truth)
    rul = (max_cycle - df["time_cycles"]) + unit_truth
    if rul_cap is not None:
        rul = rul.clip(upper=rul_cap)
    df["RUL"] = rul
    return df


def select_feature_columns(subset_name: str, drop_low_variance: bool = True) -> list[str]:
    """Sensor + setting columns to use as model features for a given subset.

    FD001/FD003 run at a single operating condition, so several sensors that only respond to
    operating condition (not degradation) are ~constant and add nothing but noise — dropped
    by default. FD002/FD004 run at 6 conditions, so those same sensors carry real signal
    there and should be kept.
    """
    cols = list(SETTING_COLS) + list(SENSOR_COLS)
    if drop_low_variance and subset_name in ("FD001", "FD003"):
        cols = [c for c in cols if c not in LOW_VARIANCE_SENSORS_SINGLE_CONDITION]
    return cols


def assign_operating_condition(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_conditions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, KMeans]:
    """Cluster the 3 operational settings into discrete condition regimes.

    FD002/FD004 advertise a fixed number of operating conditions, but the raw setting values
    have simulation noise, so exact-match grouping is unreliable — clustering recovers the
    regimes robustly and lets us normalize sensors within each one.
    """
    train_df, test_df = train_df.copy(), test_df.copy()
    if n_conditions <= 1:
        train_df["condition"] = 0
        test_df["condition"] = 0
        return train_df, test_df, None  # type: ignore[return-value]

    kmeans = KMeans(n_clusters=n_conditions, n_init=10, random_state=0)
    kmeans.fit(train_df[SETTING_COLS])
    train_df["condition"] = kmeans.predict(train_df[SETTING_COLS])
    test_df["condition"] = kmeans.predict(test_df[SETTING_COLS])
    return train_df, test_df, kmeans


def normalize_by_condition(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, StandardScaler]]:
    """Z-score each feature within each operating condition (fit on train, applied to both).

    A sensor reading of "1400 degR" means something different at sea level vs. altitude
    (different operating condition) — normalizing within condition removes that regime shift
    so the degradation trend is comparable across an engine's whole life, even for engines
    that cycle through multiple conditions (FD002/FD004).
    """
    train_df, test_df = train_df.copy(), test_df.copy()
    train_df[feature_cols] = train_df[feature_cols].astype(float)
    test_df[feature_cols] = test_df[feature_cols].astype(float)
    scalers: dict[int, StandardScaler] = {}

    for condition, group in train_df.groupby("condition"):
        scaler = StandardScaler()
        train_df.loc[group.index, feature_cols] = scaler.fit_transform(group[feature_cols])
        scalers[condition] = scaler

    for condition, group in test_df.groupby("condition"):
        scaler = scalers.get(condition)
        if scaler is None:
            # condition seen in test but not train (shouldn't happen with CMAPSS, but be safe)
            scaler = StandardScaler().fit(group[feature_cols])
            scalers[condition] = scaler
        test_df.loc[group.index, feature_cols] = scaler.transform(group[feature_cols])

    return train_df, test_df, scalers


def add_rolling_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
) -> tuple[pd.DataFrame, list[str]]:
    """Add rolling mean/std per unit for each feature — smooths sensor noise and exposes the
    degradation trend (rate of change), which a single noisy reading can't.
    """
    df = df.sort_values(["unit_number", "time_cycles"]).copy()
    grouped = df.groupby("unit_number")[feature_cols]
    new_frames: list[pd.DataFrame] = []

    for window in windows:
        mean = grouped.transform(lambda s, w=window: s.rolling(w, min_periods=1).mean())
        std = grouped.transform(
            lambda s, w=window: s.rolling(w, min_periods=1).std().fillna(0.0)
        )
        mean.columns = [f"{c}_roll_mean_{window}" for c in feature_cols]
        std.columns = [f"{c}_roll_std_{window}" for c in feature_cols]
        new_frames.extend([mean, std])

    new_cols = [c for frame in new_frames for c in frame.columns]
    df = pd.concat([df, *new_frames], axis=1)
    return df, new_cols


def compute_health_indicator(
    df: pd.DataFrame, feature_cols: list[str], degrading_sign: dict[str, int] | None = None
) -> pd.Series:
    """A single scalar Health Indicator in [0, 1] summarizing overall engine condition.

    Assumes `feature_cols` are already normalized (z-scored). Averages the normalized,
    sign-aligned sensors (flipping the sign for sensors that *decrease* with degradation) so
    that higher = more degraded, then squashes to [0, 1] and inverts so 1.0 = healthy and
    0.0 = failed. This is a simple, interpretable stand-in for the sensor-fusion "health
    indicator" that real prognostics systems derive from many correlated channels; the
    per-model RUL regressors downstream use the full feature set, not just this summary.
    """
    aligned = df[feature_cols].copy()
    if degrading_sign:
        for col, sign in degrading_sign.items():
            if col in aligned.columns:
                aligned[col] = aligned[col] * sign
    degradation_score = aligned.mean(axis=1)
    # min-max squash to [0, 1] then invert: higher degradation_score -> lower health
    lo, hi = degradation_score.min(), degradation_score.max()
    span = (hi - lo) or 1.0
    normalized = (degradation_score - lo) / span
    return 1.0 - normalized


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
    label_col: str = "RUL",
    unit_col: str = "unit_number",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide a fixed-length window over each unit's trajectory for sequence models
    (LSTM / TCN / Transformer). Units shorter than `seq_len` are left-padded by repeating
    their first row, so every unit contributes at least one sequence.

    Returns (X, y, unit_ids) where X has shape (n_sequences, seq_len, n_features).
    """
    X_list, y_list, unit_list = [], [], []

    for unit, group in df.sort_values("time_cycles").groupby(unit_col):
        features = group[feature_cols].to_numpy(dtype=np.float32)
        labels = group[label_col].to_numpy(dtype=np.float32)

        if len(group) < seq_len:
            pad = np.repeat(features[:1], seq_len - len(group), axis=0)
            features = np.concatenate([pad, features], axis=0)
            X_list.append(features)
            y_list.append(labels[-1])
            unit_list.append(unit)
            continue

        for end in range(seq_len, len(group) + 1):
            X_list.append(features[end - seq_len : end])
            y_list.append(labels[end - 1])
            unit_list.append(unit)

    return np.stack(X_list), np.array(y_list, dtype=np.float32), np.array(unit_list)
