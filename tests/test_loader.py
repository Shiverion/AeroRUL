import pytest

from aerorul.data.loader import DEFAULT_RAW_DIR, load_subset
from aerorul.data.schema import RAW_COLS, SUBSET_INFO

pytestmark = pytest.mark.skipif(
    not (DEFAULT_RAW_DIR / "train_FD001.txt").exists(),
    reason="raw CMAPSS data not downloaded; run scripts/download_data.sh",
)


def test_load_subset_shapes_match_readme():
    data = load_subset("FD001")
    info = SUBSET_INFO["FD001"]
    assert data.train["unit_number"].nunique() == info["train_units"]
    assert len(data.rul_truth) == info["test_units"]
    assert list(data.train.columns) == RAW_COLS
