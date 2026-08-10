"""Loads trained per-subset (model, pipeline) bundles and serves predictions.

Also precomputes fleet-wide predictions over each subset's CMAPSS test set at startup, so
the dashboard has real fleet data to show without needing a live sensor feed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd

from aerorul.data.loader import load_subset
from aerorul.data.schema import SENSOR_COLS, SUBSETS
from aerorul.features.pipeline import FittedPipeline, last_cycle_per_unit
from aerorul.models.baseline_xgboost import load_model, predict
from aerorul.models.decision import assess

MODELS_DIR = Path(__file__).resolve().parents[1] / "models_store"


@dataclass
class SubsetBundle:
    subset: str
    model: object
    pipeline: FittedPipeline
    fleet: pd.DataFrame  # last-cycle predictions for every test-set engine
    history: dict[int, pd.DataFrame]  # unit_number -> full raw test trajectory


class ModelRegistry:
    """Lazily-loaded, process-wide cache of trained model bundles, one per CMAPSS subset."""

    _bundles: ClassVar[dict[str, SubsetBundle]] = {}

    @classmethod
    def available_subsets(cls) -> list[str]:
        return [s for s in SUBSETS if (MODELS_DIR / f"{s}_xgboost.joblib").exists()]

    @classmethod
    def get(cls, subset: str) -> SubsetBundle:
        if subset not in cls._bundles:
            cls._bundles[subset] = cls._load(subset)
        return cls._bundles[subset]

    @classmethod
    def _load(cls, subset: str) -> SubsetBundle:
        model_path = MODELS_DIR / f"{subset}_xgboost.joblib"
        pipeline_path = MODELS_DIR / f"{subset}_pipeline.joblib"
        if not model_path.exists() or not pipeline_path.exists():
            raise FileNotFoundError(
                f"No trained model for {subset}. Run: uv run python scripts/train.py "
                f"--subset {subset}"
            )

        model = load_model(model_path)
        pipeline = FittedPipeline.load(pipeline_path)

        data = load_subset(subset)
        history = {
            unit: group.sort_values("time_cycles")
            for unit, group in data.test.groupby("unit_number")
        }

        test_features = pipeline.transform(data.test)
        fleet_rows = last_cycle_per_unit(test_features)
        preds = predict(model, fleet_rows[pipeline.all_feature_cols])
        fleet = fleet_rows[["unit_number", "time_cycles"]].copy()
        fleet["predicted_rul"] = preds
        fleet["true_rul"] = fleet["unit_number"].map(data.rul_truth).to_numpy()

        return SubsetBundle(
            subset=subset, model=model, pipeline=pipeline, fleet=fleet, history=history
        )


def predict_from_readings(subset: str, readings_df: pd.DataFrame) -> dict:
    """Run the full pipeline on caller-supplied raw sensor readings and return a prediction
    for the most recent cycle in the sequence.
    """
    bundle = ModelRegistry.get(subset)
    features = bundle.pipeline.transform(readings_df)
    latest = features.sort_values("time_cycles").tail(1)
    rul = float(predict(bundle.model, latest[bundle.pipeline.all_feature_cols])[0])
    assessment = assess(rul)
    return {
        "unit_number": int(latest["unit_number"].iloc[0]),
        "latest_cycle": int(latest["time_cycles"].iloc[0]),
        "predicted_rul": round(rul, 1),
        "risk_tier": assessment.risk_tier,
        "recommendation": assessment.recommendation,
    }


def fleet_summary(subset: str) -> list[dict]:
    bundle = ModelRegistry.get(subset)
    rows = []
    for _, row in bundle.fleet.iterrows():
        assessment = assess(row["predicted_rul"])
        rows.append(
            {
                "unit_number": int(row["unit_number"]),
                "latest_cycle": int(row["time_cycles"]),
                "predicted_rul": round(float(row["predicted_rul"]), 1),
                "true_rul": float(row["true_rul"]) if pd.notna(row["true_rul"]) else None,
                "risk_tier": assessment.risk_tier,
                "recommendation": assessment.recommendation,
            }
        )
    return sorted(rows, key=lambda r: r["predicted_rul"])


def engine_history(subset: str, unit_number: int) -> dict:
    bundle = ModelRegistry.get(subset)
    if unit_number not in bundle.history:
        raise KeyError(f"Unit {unit_number} not found in {subset} test fleet")

    raw = bundle.history[unit_number]
    fleet_row = bundle.fleet[bundle.fleet["unit_number"] == unit_number].iloc[0]
    assessment = assess(float(fleet_row["predicted_rul"]))

    return {
        "unit_number": unit_number,
        "subset": subset,
        "cycles": raw["time_cycles"].tolist(),
        "sensors": {col: raw[col].tolist() for col in SENSOR_COLS},
        "predicted_rul": round(float(fleet_row["predicted_rul"]), 1),
        "true_rul": float(fleet_row["true_rul"]) if pd.notna(fleet_row["true_rul"]) else None,
        "risk_tier": assessment.risk_tier,
        "recommendation": assessment.recommendation,
    }
