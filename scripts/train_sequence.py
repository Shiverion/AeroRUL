"""Train a sequence model (LSTM / TCN / Transformer) on a CMAPSS subset and evaluate on the
held-out test set using the standard last-cycle-per-unit protocol.

Usage:
    uv run python scripts/train_sequence.py --subset FD001 --model lstm
    uv run python scripts/train_sequence.py --subset all --model all
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from aerorul.data.schema import SUBSETS
from aerorul.evaluation.metrics import evaluate
from aerorul.features.pipeline import build_training_data
from aerorul.models.lstm import predict_lstm, save_lstm, train_lstm
from aerorul.models.sequence_common import TrainConfig, build_last_sequence_per_unit
from aerorul.models.tcn import predict_tcn, save_tcn, train_tcn
from aerorul.models.transformer import predict_transformer, save_transformer, train_transformer

MODELS_DIR = Path(__file__).resolve().parents[1] / "models_store"

TRAINERS = {
    "lstm": (train_lstm, predict_lstm, save_lstm),
    "tcn": (train_tcn, predict_tcn, save_tcn),
    "transformer": (train_transformer, predict_transformer, save_transformer),
}


def run(subset: str, model_type: str, config: TrainConfig, models_dir: Path = MODELS_DIR) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df, pipeline, data = build_training_data(subset)
    feature_cols = pipeline.base_feature_cols  # sequence models learn temporal patterns themselves

    # y_test from build_last_sequence_per_unit is the RUL-capped training label; evaluation
    # must use the true, uncapped RUL from RUL_*.txt (see the comment in scripts/train.py).
    X_test, _, units_test = build_last_sequence_per_unit(test_df, feature_cols, config.seq_len)
    y_test = pd.Series(units_test).map(data.rul_truth).to_numpy()

    trainer, predictor, saver = TRAINERS[model_type]

    with mlflow.start_run(run_name=f"{model_type}_{subset}"):
        mlflow.log_params(
            {
                "subset": subset,
                "model_type": model_type,
                "seq_len": config.seq_len,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "lr": config.lr,
                "n_features": len(feature_cols),
            }
        )

        start = time.time()
        model, history = trainer(train_df, feature_cols, config)
        train_seconds = time.time() - start

        y_pred = predictor(model, X_test, config.rul_cap)
        metrics = evaluate(y_test, y_pred)
        metrics["train_seconds"] = train_seconds
        metrics["epochs_run"] = len(history["train_loss"])
        mlflow.log_metrics(
            {k: v for k, v in metrics.items() if k not in ("epochs_run",)}
        )
        mlflow.log_metric("epochs_run", metrics["epochs_run"])

        weights_path = models_dir / f"{subset}_{model_type}.pt"
        meta_path = models_dir / f"{subset}_{model_type}_meta.joblib"
        saver(model, config, str(weights_path))
        joblib.dump(
            {
                "feature_cols": feature_cols,
                "seq_len": config.seq_len,
                "rul_cap": config.rul_cap,
                "n_features": len(feature_cols),
            },
            meta_path,
        )
        mlflow.log_artifact(str(weights_path))
        mlflow.log_artifact(str(meta_path))

    print(
        f"[{subset}/{model_type}] test RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  "
        f"NASA score={metrics['nasa_score']:.1f}  epochs={metrics['epochs_run']}  "
        f"({train_seconds:.0f}s, n={len(y_test)} engines)"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD001", choices=[*SUBSETS, "all"])
    parser.add_argument("--model", default="lstm", choices=[*TRAINERS, "all"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seq-len", type=int, default=30)
    args = parser.parse_args()

    mlflow.set_experiment("aerorul-rul-prediction")
    subsets = SUBSETS if args.subset == "all" else [args.subset]
    model_types = list(TRAINERS) if args.model == "all" else [args.model]
    config = TrainConfig(seq_len=args.seq_len, epochs=args.epochs)

    for subset in subsets:
        for model_type in model_types:
            run(subset, model_type, config)


if __name__ == "__main__":
    main()
