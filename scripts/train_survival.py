"""Fit the Weibull AFT survival model on a CMAPSS subset and evaluate expected-RUL
predictions on the real (right-censored) test set, using the same metrics as the other
models for direct comparison.

Usage:
    uv run python scripts/train_survival.py --subset all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mlflow

from aerorul.data.schema import SUBSETS
from aerorul.evaluation.metrics import evaluate
from aerorul.features.pipeline import build_training_data
from aerorul.models.survival import (
    DEFAULT_LANDMARK_STRIDE,
    build_censored_frame,
    build_landmark_frame,
    predict_median_rul,
    train_survival_model,
)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models_store"


def run(subset: str, stride: int = DEFAULT_LANDMARK_STRIDE, models_dir: Path = MODELS_DIR) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df, pipeline, data = build_training_data(subset)
    feature_cols = pipeline.base_feature_cols

    landmark_frame = build_landmark_frame(train_df, feature_cols, stride=stride)
    censored_frame = build_censored_frame(test_df, feature_cols)

    with mlflow.start_run(run_name=f"survival_{subset}"):
        mlflow.log_params(
            {
                "subset": subset,
                "model_type": "weibull_aft",
                "landmark_stride": stride,
                "n_landmarks": len(landmark_frame),
            }
        )

        model = train_survival_model(landmark_frame)
        mlflow.log_metric("train_concordance_index", model.concordance_index_)

        y_pred = predict_median_rul(model, censored_frame[feature_cols]).to_numpy()
        y_true = censored_frame.index.to_series().map(data.rul_truth).to_numpy()

        metrics = evaluate(y_true, y_pred)
        mlflow.log_metrics(metrics)

        path = models_dir / f"{subset}_survival.joblib"
        joblib.dump({"model": model, "feature_cols": feature_cols}, path)
        mlflow.log_artifact(str(path))

    print(
        f"[{subset}/survival] test RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  "
        f"NASA score={metrics['nasa_score']:.1f}  train c-index={model.concordance_index_:.3f}  "
        f"(n_landmarks={len(landmark_frame)}, n_test={len(y_true)} engines)"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD001", choices=[*SUBSETS, "all"])
    parser.add_argument("--stride", type=int, default=DEFAULT_LANDMARK_STRIDE)
    args = parser.parse_args()

    mlflow.set_experiment("aerorul-rul-prediction")
    subsets = SUBSETS if args.subset == "all" else [args.subset]
    for subset in subsets:
        run(subset, args.stride)


if __name__ == "__main__":
    main()
