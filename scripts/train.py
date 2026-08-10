"""Train the XGBoost RUL baseline on a CMAPSS subset and evaluate on the held-out test set.

Usage:
    uv run python scripts/train.py --subset FD001
    uv run python scripts/train.py --subset all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow

from aerorul.data.schema import SUBSETS
from aerorul.evaluation.metrics import evaluate
from aerorul.features.pipeline import build_training_data, last_cycle_per_unit
from aerorul.models.baseline_xgboost import DEFAULT_PARAMS, predict, save_model, train_xgboost

MODELS_DIR = Path(__file__).resolve().parents[1] / "models_store"


def run(subset: str, models_dir: Path = MODELS_DIR) -> dict[str, float]:
    models_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df, pipeline, _ = build_training_data(subset)
    test_last = last_cycle_per_unit(test_df)

    X_train, y_train = train_df[pipeline.all_feature_cols], train_df["RUL"]
    X_test, y_test = test_last[pipeline.all_feature_cols], test_last["RUL"]

    with mlflow.start_run(run_name=f"xgboost_{subset}"):
        mlflow.log_params({**DEFAULT_PARAMS, "subset": subset, "rul_cap": pipeline.rul_cap})
        mlflow.log_param("n_features", len(pipeline.all_feature_cols))

        model = train_xgboost(X_train, y_train)
        y_pred = predict(model, X_test)
        metrics = evaluate(y_test.to_numpy(), y_pred)
        mlflow.log_metrics(metrics)

        model_path = models_dir / f"{subset}_xgboost.joblib"
        pipeline_path = models_dir / f"{subset}_pipeline.joblib"
        save_model(model, model_path)
        pipeline.save(pipeline_path)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(pipeline_path))

    print(f"[{subset}] test RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  "
          f"NASA score={metrics['nasa_score']:.1f}  (n={len(y_test)} engines)")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD001", choices=[*SUBSETS, "all"])
    args = parser.parse_args()

    mlflow.set_experiment("aerorul-rul-prediction")
    subsets = SUBSETS if args.subset == "all" else [args.subset]
    for subset in subsets:
        run(subset)


if __name__ == "__main__":
    main()
