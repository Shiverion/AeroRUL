"""Evaluate every trained model (XGBoost, LSTM, TCN, Transformer, survival) on each CMAPSS
subset's real test set with the same metrics, then pick a per-subset champion by NASA score
(the metric that actually reflects maintenance-decision cost, not just point-estimate error).

Writes models_store/champion.json, which the API reads at startup to decide which model to
serve for each subset -- this is the "then deploy the winner" step.

Usage:
    uv run python scripts/compare_models.py --subset all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from aerorul.data.schema import SUBSETS
from aerorul.evaluation.metrics import evaluate
from aerorul.features.pipeline import build_training_data, last_cycle_per_unit
from aerorul.models.baseline_xgboost import load_model
from aerorul.models.baseline_xgboost import predict as predict_xgb
from aerorul.models.lstm import load_lstm, predict_lstm
from aerorul.models.sequence_common import build_last_sequence_per_unit
from aerorul.models.survival import build_censored_frame, predict_median_rul
from aerorul.models.tcn import load_tcn, predict_tcn
from aerorul.models.transformer import load_transformer, predict_transformer

MODELS_DIR = Path(__file__).resolve().parents[1] / "models_store"

SEQUENCE_LOADERS = {
    "lstm": (load_lstm, predict_lstm),
    "tcn": (load_tcn, predict_tcn),
    "transformer": (load_transformer, predict_transformer),
}


def evaluate_subset(subset: str, models_dir: Path = MODELS_DIR) -> dict[str, dict]:
    _train_df, test_df, pipeline, data = build_training_data(subset)
    y_true_by_unit = data.rul_truth

    results: dict[str, dict] = {}

    xgb_path = models_dir / f"{subset}_xgboost.joblib"
    if xgb_path.exists():
        model = load_model(xgb_path)
        test_last = last_cycle_per_unit(test_df)
        y_pred = predict_xgb(model, test_last[pipeline.all_feature_cols])
        y_true = test_last["unit_number"].map(y_true_by_unit).to_numpy()
        results["xgboost"] = evaluate(y_true, y_pred)

    for model_type, (loader, predictor) in SEQUENCE_LOADERS.items():
        weights_path = models_dir / f"{subset}_{model_type}.pt"
        meta_path = models_dir / f"{subset}_{model_type}_meta.joblib"
        if not (weights_path.exists() and meta_path.exists()):
            continue
        meta = joblib.load(meta_path)
        model, _config = loader(str(weights_path), meta["n_features"])
        # y_test here would be the RUL-capped training label; use the true, uncapped RUL
        # from RUL_*.txt instead, same reasoning as the XGBoost branch above.
        X_test, _capped_y, units_test = build_last_sequence_per_unit(
            test_df, meta["feature_cols"], meta["seq_len"]
        )
        y_true = pd.Series(units_test).map(y_true_by_unit).to_numpy()
        y_pred = predictor(model, X_test, meta["rul_cap"])
        results[model_type] = evaluate(y_true, y_pred)

    survival_path = models_dir / f"{subset}_survival.joblib"
    if survival_path.exists():
        bundle = joblib.load(survival_path)
        censored_frame = build_censored_frame(test_df, bundle["feature_cols"])
        y_pred = predict_median_rul(bundle["model"], censored_frame[bundle["feature_cols"]]).to_numpy()
        y_true = censored_frame.index.to_series().map(y_true_by_unit).to_numpy()
        results["survival"] = evaluate(y_true, y_pred)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="all", choices=[*SUBSETS, "all"])
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    args = parser.parse_args()
    models_dir = Path(args.models_dir)

    subsets = SUBSETS if args.subset == "all" else [args.subset]
    champions: dict[str, dict] = {}

    for subset in subsets:
        results = evaluate_subset(subset, models_dir)
        if not results:
            print(f"[{subset}] no trained models found, skipping")
            continue

        print(f"\n{subset}")
        print(f"{'model':<14}{'rmse':>8}{'mae':>8}{'nasa_score':>14}")
        for model_type, metrics in sorted(results.items(), key=lambda kv: kv[1]["nasa_score"]):
            print(
                f"{model_type:<14}{metrics['rmse']:>8.2f}{metrics['mae']:>8.2f}"
                f"{metrics['nasa_score']:>14.1f}"
            )

        champion = min(results, key=lambda m: results[m]["nasa_score"])
        champions[subset] = {"champion": champion, "metrics": results}
        print(f"-> champion: {champion} (lowest NASA score)")

    champion_path = models_dir / "champion.json"
    champion_path.write_text(json.dumps(champions, indent=2))
    print(f"\nWrote {champion_path}")


if __name__ == "__main__":
    main()
