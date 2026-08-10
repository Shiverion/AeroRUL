"""Calibrate split-conformal prediction intervals for the XGBoost baseline on each subset.

Trains on 80% of train-set engines, calibrates residuals on the held-out 20% (every cycle,
not just last-cycle, since a live prediction can be requested at any point in an engine's
life), then reports empirical coverage on the real CMAPSS test set as a sanity check that
the guarantee actually holds out-of-sample.

Usage:
    uv run python scripts/calibrate_uncertainty.py --subset all --alpha 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from aerorul.data.schema import SUBSETS
from aerorul.features.engineering import add_rul_train
from aerorul.features.pipeline import build_training_data, last_cycle_per_unit
from aerorul.models.baseline_xgboost import predict, train_xgboost
from aerorul.models.sequence_common import train_val_split_by_unit
from aerorul.uncertainty.conformal import ConformalIntervals, evaluate_coverage

MODELS_DIR = Path(__file__).resolve().parents[1] / "models_store"


def run(subset: str, alpha: float, models_dir: Path = MODELS_DIR) -> ConformalIntervals:
    train_df, test_df, pipeline, data = build_training_data(subset)
    train_df = train_df.copy()
    train_df["RUL_uncapped"] = add_rul_train(data.train, rul_cap=None)["RUL"]
    fit_df, cal_df = train_val_split_by_unit(train_df, val_fraction=0.2, seed=0)

    # Calibration uses the RUL-capped label, matching what the model was trained to predict.
    # But it must exclude rows whose *true* RUL exceeds the cap (healthy, early-life cycles)
    # -- those rows' capped label of 125 isn't really "the answer", it's a ceiling the model
    # was never asked to see past, so including them would calibrate against a target the
    # model structurally can't hit and silently under-cover the near-failure predictions
    # that actually drive maintenance decisions.
    cal_df = cal_df[cal_df["RUL_uncapped"] <= (pipeline.rul_cap or float("inf"))]
    model = train_xgboost(fit_df[pipeline.all_feature_cols], fit_df["RUL"])
    cal_pred = predict(model, cal_df[pipeline.all_feature_cols])
    intervals = ConformalIntervals.calibrate(cal_df["RUL"].to_numpy(), cal_pred, alpha=alpha)

    test_last = last_cycle_per_unit(test_df)
    test_pred = predict(model, test_last[pipeline.all_feature_cols])
    lower, upper = intervals.interval(test_pred)
    y_true = test_last["unit_number"].map(data.rul_truth).to_numpy()
    coverage = evaluate_coverage(y_true, lower, upper)

    path = models_dir / f"{subset}_conformal.joblib"
    joblib.dump(intervals, path)

    print(
        f"[{subset}] target coverage={intervals.coverage_target:.0%}  "
        f"empirical={coverage['empirical_coverage']:.0%}  "
        f"mean width=+/-{intervals.q:.1f} cycles"
    )
    return intervals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD001", choices=[*SUBSETS, "all"])
    parser.add_argument("--alpha", type=float, default=0.1, help="miscoverage rate (0.1 = 90% target coverage)")
    args = parser.parse_args()

    subsets = SUBSETS if args.subset == "all" else [args.subset]
    for subset in subsets:
        run(subset, args.alpha)


if __name__ == "__main__":
    main()
