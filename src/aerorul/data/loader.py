"""Load raw CMAPSS train/test/RUL text files into pandas DataFrames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from aerorul.data.schema import RAW_COLS, SUBSETS

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"


@dataclass(frozen=True)
class CMAPSSSubset:
    """Train/test/RUL data for one CMAPSS subset (e.g. FD001)."""

    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    rul_truth: pd.Series  # true RUL at the last cycle of each test unit, indexed by unit_number


def _read_raw_txt(path: Path) -> pd.DataFrame:
    """CMAPSS txt files are whitespace-separated with two trailing blank columns."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(RAW_COLS)]
    df.columns = RAW_COLS
    df["unit_number"] = df["unit_number"].astype(int)
    df["time_cycles"] = df["time_cycles"].astype(int)
    return df


def load_subset(subset: str, raw_dir: Path | str = DEFAULT_RAW_DIR) -> CMAPSSSubset:
    """Load one CMAPSS subset (FD001-FD004) from raw text files."""
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}, expected one of {SUBSETS}")
    raw_dir = Path(raw_dir)

    train = _read_raw_txt(raw_dir / f"train_{subset}.txt")
    test = _read_raw_txt(raw_dir / f"test_{subset}.txt")

    rul_path = raw_dir / f"RUL_{subset}.txt"
    rul_values = pd.read_csv(rul_path, header=None).iloc[:, 0].astype(int)
    rul_truth = pd.Series(
        rul_values.values,
        index=pd.RangeIndex(1, len(rul_values) + 1, name="unit_number"),
        name="RUL",
    )

    return CMAPSSSubset(name=subset, train=train, test=test, rul_truth=rul_truth)


def load_all_subsets(raw_dir: Path | str = DEFAULT_RAW_DIR) -> dict[str, CMAPSSSubset]:
    return {subset: load_subset(subset, raw_dir) for subset in SUBSETS}
