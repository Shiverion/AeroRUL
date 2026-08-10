import numpy as np

from aerorul.evaluation.metrics import evaluate, nasa_score, rmse


def test_rmse_zero_for_perfect_predictions():
    y = np.array([10.0, 20.0, 30.0])
    assert rmse(y, y) == 0.0


def test_nasa_score_penalizes_late_predictions_more_than_early():
    y_true = np.array([50.0])
    early = nasa_score(y_true, np.array([40.0]))  # predicted 10 cycles too low
    late = nasa_score(y_true, np.array([60.0]))  # predicted 10 cycles too high
    assert late > early


def test_evaluate_bundle_has_expected_keys():
    result = evaluate(np.array([10.0, 20.0]), np.array([12.0, 18.0]))
    assert set(result) == {"rmse", "mae", "nasa_score"}
