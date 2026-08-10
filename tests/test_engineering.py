import pandas as pd
import pytest

from aerorul.data.schema import SENSOR_COLS, SETTING_COLS
from aerorul.features.engineering import add_rul_test, add_rul_train, build_sequences


def _synthetic_unit(unit_number: int, n_cycles: int) -> pd.DataFrame:
    data = {"unit_number": unit_number, "time_cycles": range(1, n_cycles + 1)}
    for col in SETTING_COLS + SENSOR_COLS:
        data[col] = [float(i) for i in range(n_cycles)]
    return pd.DataFrame(data)


@pytest.fixture
def train_df():
    return pd.concat([_synthetic_unit(1, 20), _synthetic_unit(2, 10)], ignore_index=True)


def test_add_rul_train_counts_down_to_zero_at_last_cycle(train_df):
    labeled = add_rul_train(train_df, rul_cap=None)
    last_rows = labeled.groupby("unit_number").tail(1)
    assert (last_rows["RUL"] == 0).all()


def test_add_rul_train_caps_at_rul_cap(train_df):
    labeled = add_rul_train(train_df, rul_cap=5)
    assert labeled["RUL"].max() == 5


def test_add_rul_test_uses_truth_at_truncation_point(train_df):
    rul_truth = pd.Series({1: 15, 2: 3})
    labeled = add_rul_test(train_df, rul_truth, rul_cap=None)
    last_rows = labeled.groupby("unit_number").tail(1).set_index("unit_number")
    assert last_rows.loc[1, "RUL"] == 15
    assert last_rows.loc[2, "RUL"] == 3


def test_build_sequences_pads_short_units(train_df):
    labeled = add_rul_train(train_df, rul_cap=None)
    feature_cols = SETTING_COLS + SENSOR_COLS
    X, y, units = build_sequences(labeled, feature_cols, seq_len=15)
    # unit 2 has only 10 cycles < seq_len=15 -> exactly one padded sequence
    assert (units == 2).sum() == 1
    # unit 1 has 20 cycles >= seq_len=15 -> 20 - 15 + 1 sliding windows
    assert (units == 1).sum() == 6
    assert X.shape[1:] == (15, len(feature_cols))
    assert len(y) == len(units)
