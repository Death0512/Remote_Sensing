from datetime import datetime, timezone
from src.core.geofence import GeofenceManager
from src.detectors.change_detector import DualStreamChangeDetector
from src.engine.anomaly_scorer import AnomalyScorer
from tests.mocks.mock_data import generate_synthetic_bitemporal_pair


def test_change_pipeline_end_to_end_scoring():
    t1, t2 = generate_synthetic_bitemporal_pair(
        height=100,
        width=100,
        new_structures=[(50, 50, 4, 4, 0.9)],
    )

    detector = DualStreamChangeDetector(
        weights_config_path="configs/model_registry.yaml",
        pixel_spacing_m=10.0,
        diff_threshold=0.3,
        min_area_px=15,
    )

    transform = (0.001, 0.0, 108.5, 0.0, -0.001, 16.5)
    acq_time = datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc)

    detections = detector.detect(
        (t1, t2),
        sensor="S1_RTC",
        source_tile="REEF_OUTPOST_MONITOR",
        acquisition_time=acq_time,
        transform_matrix=transform,
    )

    assert len(detections) >= 1
    det = detections[0]
    assert det.target_class == "unknown_structure"

    geofence = GeofenceManager(center_lat=16.0, center_lon=108.0)
    centroid = det.geom_wgs84.centroid
    zone = geofence.classify_zone(centroid.y, centroid.x)

    scorer = AnomalyScorer(
        weights={
            "w_z": 0.40,
            "w_d": 0.20,
            "w_v": 0.10,
            "w_c": 0.20,
            "w_b": 0.10,
        },
        reference_distance_km=500.0,
    )

    dist_km = geofence.compute_geodesic_distance_m(centroid.y, centroid.x) / 1000.0
    score = scorer.calculate_score(
        zone=zone,
        is_dark=None,
        velocity_score=0.0,
        change_score=1.0,
        geodesic_distance_km=dist_km,
    )

    assert 0.0 <= score <= 100.0
    assert score >= 50.0
