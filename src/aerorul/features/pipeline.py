"""Ties together loading + RUL labeling + condition normalization + rolling features into
one reusable feature-engineering pipeline, with a fitted-artifact bundle that inference
(the API) can replay exactly on new engine data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from aerorul.data.loader import CMAPSSSubset, load_subset
from aerorul.data.schema import SUBSET_INFO
from aerorul.features.engineering import (
    DEFAULT_ROLLING_WINDOWS,
    DEFAULT_RUL_CAP,
    add_rolling_features,
    add_rul_test,
    add_rul_train,
    assign_operating_condition,
    normalize_by_condition,
    select_feature_columns,
)


@dataclass
class FittedPipeline:
    """Everything needed to reproduce feature engineering on new, unseen engine data."""

    subset: str
    base_feature_cols: list[str]
    all_feature_cols: list[str]  # base + rolling stats, in the exact order the model expects
    rolling_windows: tuple[int, ...]
    rul_cap: int | None
    kmeans: KMeans | None
    scalers: dict[int, StandardScaler]

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted condition clustering, normalization, and rolling features to a
        raw (unlabeled) dataframe of engine-cycle rows, e.g. live sensor data at inference.
        """
        df = raw_df.copy()
        df[self.base_feature_cols] = df[self.base_feature_cols].astype(float)

        if self.kmeans is not None:
            df["condition"] = self.kmeans.predict(df[["setting_1", "setting_2", "setting_3"]])
        else:
            df["condition"] = 0

        for condition, group in df.groupby("condition"):
            scaler = self.scalers.get(condition)
            if scaler is not None:
                df.loc[group.index, self.base_feature_cols] = scaler.transform(
                    group[self.base_feature_cols]
                )

        df, _ = add_rolling_features(df, self.base_feature_cols, self.rolling_windows)
        return df

    def save(self, path: Path | str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path | str) -> FittedPipeline:
        return joblib.load(path)


def build_training_data(
    subset: str,
    rul_cap: int | None = DEFAULT_RUL_CAP,
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
    raw_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, FittedPipeline, CMAPSSSubset]:
    """Load raw CMAPSS data for a subset and run the full feature pipeline, fit on train.

    Returns (train_features, test_features, fitted_pipeline, raw_subset). test_features
    covers every test cycle (needed for sequence models); for the tabular baseline, callers
    should evaluate only on each unit's last cycle to match the CMAPSS test protocol.
    """
    kwargs = {} if raw_dir is None else {"raw_dir": raw_dir}
    data = load_subset(subset, **kwargs)

    train = add_rul_train(data.train, rul_cap=rul_cap)
    test = add_rul_test(data.test, data.rul_truth, rul_cap=rul_cap)

    base_feature_cols = select_feature_columns(subset)
    n_conditions = SUBSET_INFO[subset]["conditions"]
    train, test, kmeans = assign_operating_condition(train, test, n_conditions)
    train, test, scalers = normalize_by_condition(train, test, base_feature_cols)
    train, roll_cols = add_rolling_features(train, base_feature_cols, rolling_windows)
    test, _ = add_rolling_features(test, base_feature_cols, rolling_windows)

    all_feature_cols = base_feature_cols + roll_cols
    pipeline = FittedPipeline(
        subset=subset,
        base_feature_cols=base_feature_cols,
        all_feature_cols=all_feature_cols,
        rolling_windows=rolling_windows,
        rul_cap=rul_cap,
        kmeans=kmeans,
        scalers=scalers,
    )
    return train, test, pipeline, data


def last_cycle_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce a full test trajectory to one row per unit (its most recent cycle) — this is
    the standard CMAPSS test protocol: predict RUL as of the last available reading.
    """
    return df.sort_values("time_cycles").groupby("unit_number", as_index=False).tail(1)
