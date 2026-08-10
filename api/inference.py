"""Loads each subset's champion model (whichever architecture won the comparison in
scripts/compare_models.py, or XGBoost by default if no comparison has run yet) and serves
predictions, with an approximate uncertainty band from split-conformal calibration.

Also precomputes fleet-wide predictions over each subset's CMAPSS test set at startup, so
the dashboard has real fleet data to show without needing a live sensor feed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import joblib
import numpy as np
import pandas as pd

from aerorul.data.loader import load_subset
from aerorul.data.schema import SENSOR_COLS, SUBSETS
from aerorul.features.pipeline import FittedPipeline, last_cycle_per_unit
from aerorul.models.baseline_xgboost import load_model
from aerorul.models.baseline_xgboost import predict as predict_xgb
from aerorul.models.decision import assess
from aerorul.models.lstm import load_lstm, predict_lstm
from aerorul.models.sequence_common import build_last_sequence_per_unit
from aerorul.models.survival import build_censored_frame, predict_median_rul
from aerorul.models.tcn import load_tcn, predict_tcn
from aerorul.models.transformer import load_transformer, predict_transformer
from aerorul.uncertainty.conformal import ConformalIntervals

MODELS_DIR = Path(__file__).resolve().parents[1] / "models_store"

SEQUENCE_LOADERS = {
    "lstm": (load_lstm, predict_lstm),
    "tcn": (load_tcn, predict_tcn),
    "transformer": (load_transformer, predict_transformer),
}


@dataclass
class SubsetBundle:
    subset: str
    pipeline: FittedPipeline
    champion_type: str
    champion_artifact: dict
    conformal: ConformalIntervals | None
    fleet: pd.DataFrame  # last-cycle predictions (+ interval) for every test-set engine
    history: dict[int, pd.DataFrame]  # unit_number -> full raw test trajectory


def _load_champion_artifact(subset: str, champion_type: str, models_dir: Path) -> dict:
    if champion_type == "xgboost":
        return {"model": load_model(models_dir / f"{subset}_xgboost.joblib")}

    if champion_type in SEQUENCE_LOADERS:
        loader, predictor = SEQUENCE_LOADERS[champion_type]
        meta = joblib.load(models_dir / f"{subset}_{champion_type}_meta.joblib")
        model, _ = loader(str(models_dir / f"{subset}_{champion_type}.pt"), meta["n_features"])
        return {"model": model, "meta": meta, "predictor": predictor}

    if champion_type == "survival":
        bundle = joblib.load(models_dir / f"{subset}_survival.joblib")
        return {"model": bundle["model"], "feature_cols": bundle["feature_cols"]}

    raise ValueError(f"Unknown champion model type {champion_type!r}")


def _predict_with_artifact(
    champion_type: str, artifact: dict, pipeline: FittedPipeline, engineered_df: pd.DataFrame
):
    """Dispatches to whichever model type won the comparison. Returns (unit_numbers,
    latest_cycles, predicted_rul) arrays, aligned by position.
    """
    if "RUL" not in engineered_df.columns:
        engineered_df = engineered_df.assign(RUL=0.0)  # unused placeholder for inference-only calls
    last_rows = last_cycle_per_unit(engineered_df).set_index("unit_number")

    if champion_type == "xgboost":
        preds = predict_xgb(artifact["model"], last_rows[pipeline.all_feature_cols])
        units = last_rows.index.to_numpy()
    elif champion_type in SEQUENCE_LOADERS:
        meta = artifact["meta"]
        X, _, units = build_last_sequence_per_unit(engineered_df, meta["feature_cols"], meta["seq_len"])
        preds = artifact["predictor"](artifact["model"], X, meta["rul_cap"])
    elif champion_type == "survival":
        censored = build_censored_frame(engineered_df, artifact["feature_cols"])
        preds = predict_median_rul(artifact["model"], censored[artifact["feature_cols"]]).to_numpy()
        units = censored.index.to_numpy()
    else:
        raise ValueError(f"Unknown champion model type {champion_type!r}")

    cycles = last_rows.loc[units, "time_cycles"].to_numpy()
    return units, cycles, preds


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
        pipeline_path = MODELS_DIR / f"{subset}_pipeline.joblib"
        if not pipeline_path.exists():
            raise FileNotFoundError(
                f"No trained model for {subset}. Run: uv run python scripts/train.py "
                f"--subset {subset}"
            )
        pipeline = FittedPipeline.load(pipeline_path)

        champion_type = "xgboost"
        champion_path = MODELS_DIR / "champion.json"
        if champion_path.exists():
            champions = json.loads(champion_path.read_text())
            if subset in champions:
                champion_type = champions[subset]["champion"]
        champion_artifact = _load_champion_artifact(subset, champion_type, MODELS_DIR)

        conformal_path = MODELS_DIR / f"{subset}_conformal.joblib"
        conformal = joblib.load(conformal_path) if conformal_path.exists() else None

        data = load_subset(subset)
        history = {
            unit: group.sort_values("time_cycles")
            for unit, group in data.test.groupby("unit_number")
        }

        engineered_test = pipeline.transform(data.test)
        units, cycles, preds = _predict_with_artifact(
            champion_type, champion_artifact, pipeline, engineered_test
        )
        fleet = pd.DataFrame({"unit_number": units, "time_cycles": cycles, "predicted_rul": preds})
        fleet["true_rul"] = fleet["unit_number"].map(data.rul_truth).to_numpy()
        if conformal is not None:
            lower, upper = conformal.interval(preds)
            fleet["predicted_rul_lower"], fleet["predicted_rul_upper"] = lower, upper
        else:
            fleet["predicted_rul_lower"], fleet["predicted_rul_upper"] = None, None

        return SubsetBundle(
            subset=subset,
            pipeline=pipeline,
            champion_type=champion_type,
            champion_artifact=champion_artifact,
            conformal=conformal,
            fleet=fleet,
            history=history,
        )


def _interval_fields(bundle: SubsetBundle, point_pred: float) -> dict:
    if bundle.conformal is None:
        return {"predicted_rul_lower": None, "predicted_rul_upper": None}

    lower, upper = bundle.conformal.interval(np.array([point_pred]))
    return {"predicted_rul_lower": round(float(lower[0]), 1), "predicted_rul_upper": round(float(upper[0]), 1)}


def predict_from_readings(subset: str, readings_df: pd.DataFrame) -> dict:
    """Run the full pipeline on caller-supplied raw sensor readings and return a prediction
    for the most recent cycle in the sequence, using the subset's champion model.
    """
    bundle = ModelRegistry.get(subset)
    engineered = bundle.pipeline.transform(readings_df)
    units, cycles, preds = _predict_with_artifact(
        bundle.champion_type, bundle.champion_artifact, bundle.pipeline, engineered
    )
    rul = float(preds[0])
    assessment = assess(rul)
    return {
        "unit_number": int(units[0]),
        "latest_cycle": int(cycles[0]),
        "predicted_rul": round(rul, 1),
        "risk_tier": assessment.risk_tier,
        "recommendation": assessment.recommendation,
        "model_used": bundle.champion_type,
        **_interval_fields(bundle, rul),
    }


def model_comparison(subset: str) -> dict | None:
    """The per-model metrics table from the last `scripts/compare_models.py` run, plus
    which one was picked as champion -- None if that comparison hasn't been run yet.
    """
    champion_path = MODELS_DIR / "champion.json"
    if not champion_path.exists():
        return None
    champions = json.loads(champion_path.read_text())
    return champions.get(subset)


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
                "predicted_rul_lower": (
                    round(float(row["predicted_rul_lower"]), 1)
                    if pd.notna(row["predicted_rul_lower"])
                    else None
                ),
                "predicted_rul_upper": (
                    round(float(row["predicted_rul_upper"]), 1)
                    if pd.notna(row["predicted_rul_upper"])
                    else None
                ),
                "true_rul": float(row["true_rul"]) if pd.notna(row["true_rul"]) else None,
                "risk_tier": assessment.risk_tier,
                "recommendation": assessment.recommendation,
                "model_used": bundle.champion_type,
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
        "predicted_rul_lower": (
            round(float(fleet_row["predicted_rul_lower"]), 1)
            if pd.notna(fleet_row["predicted_rul_lower"])
            else None
        ),
        "predicted_rul_upper": (
            round(float(fleet_row["predicted_rul_upper"]), 1)
            if pd.notna(fleet_row["predicted_rul_upper"])
            else None
        ),
        "true_rul": float(fleet_row["true_rul"]) if pd.notna(fleet_row["true_rul"]) else None,
        "risk_tier": assessment.risk_tier,
        "recommendation": assessment.recommendation,
        "model_used": bundle.champion_type,
    }
