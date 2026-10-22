import pytest
from src.engine.anomaly_scorer import AnomalyScorer


def test_anomaly_scorer_weights_validation():
    invalid_weights = {"w_z": 0.5, "w_d": 0.2, "w_v": 0.1, "w_c": 0.1, "w_b": 0.05}
    with pytest.raises(ValueError):
        AnomalyScorer(invalid_weights)


def test_anomaly_scorer_range():
    weights = {"w_z": 0.40, "w_d": 0.25, "w_v": 0.15, "w_c": 0.10, "w_b": 0.10}
    scorer = AnomalyScorer(weights, reference_distance_km=500.0)

    score_max = scorer.calculate_score(zone=1, is_dark=True, velocity_score=1.0, change_score=1.0, geodesic_distance_km=0.0)
    assert abs(score_max - 100.0) < 1e-4

    score_min = scorer.calculate_score(zone=0, is_dark=False, velocity_score=0.0, change_score=0.0, geodesic_distance_km=1000.0)
    assert abs(score_min - 0.0) < 1e-4

    score_mid = scorer.calculate_score(zone=2, is_dark=None, velocity_score=0.5, change_score=0.2, geodesic_distance_km=250.0)
    assert 0.0 <= score_mid <= 100.0
