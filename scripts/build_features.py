"""Run the feature pipeline for a CMAPSS subset and cache the result as processed parquet
files -- lets notebooks and ad-hoc analysis load engineered features without re-running
KMeans/scaling/rolling-window computation every time.

Usage:
    uv run python scripts/build_features.py --subset all
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aerorul.data.schema import SUBSETS
from aerorul.features.pipeline import build_training_data

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def run(subset: str, processed_dir: Path = PROCESSED_DIR) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df, pipeline, data = build_training_data(subset)

    train_df.to_parquet(processed_dir / f"{subset}_train.parquet", index=False)
    test_df.to_parquet(processed_dir / f"{subset}_test.parquet", index=False)
    data.rul_truth.to_frame().to_parquet(processed_dir / f"{subset}_rul_truth.parquet")
    pipeline.save(processed_dir / f"{subset}_pipeline.joblib")

    print(
        f"[{subset}] wrote {len(train_df)} train rows, {len(test_df)} test rows, "
        f"{len(pipeline.all_feature_cols)} features -> {processed_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD001", choices=[*SUBSETS, "all"])
    args = parser.parse_args()

    subsets = SUBSETS if args.subset == "all" else [args.subset]
    for subset in subsets:
        run(subset)


if __name__ == "__main__":
    main()
