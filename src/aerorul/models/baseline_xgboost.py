"""XGBoost baseline: tabular regression from engineered features straight to RUL."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 0,
    "n_jobs": -1,
}


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**{**DEFAULT_PARAMS, **(params or {})})
    model.fit(X_train, y_train)
    return model


def predict(model: xgb.XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    preds = model.predict(X)
    return np.clip(preds, a_min=0, a_max=None)  # RUL can't be negative


def save_model(model: xgb.XGBRegressor, path: Path | str) -> None:
    joblib.dump(model, path)


def load_model(path: Path | str) -> xgb.XGBRegressor:
    return joblib.load(path)
