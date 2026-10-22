from datetime import datetime, timezone
from src.detectors.parallax_detector import ParallaxInflightDetector
from tests.mocks.mock_data import generate_synthetic_parallax_scene


def test_parallax_detector_inflight_triplet():
    scene = generate_synthetic_parallax_scene(
        height=100,
        width=100,
        triplets=[
            ((40, 40), (43, 44), (46, 48)),
        ],
    )

    detector = ParallaxInflightDetector(
        pixel_spacing_m=10.0,
        inter_band_delay_seconds=0.52,
        min_velocity_kmh=300.0,
        max_velocity_kmh=1000.0,
    )

    transform = (0.0001, 0.0, 108.0, 0.0, -0.0001, 16.0)
    acq_time = datetime(2026, 10, 22, 11, 0, 0, tzinfo=timezone.utc)

    detections = detector.detect(
        scene,
        sensor="S2_L1C",
        source_tile="S2A_TEST_TILE",
        acquisition_time=acq_time,
        transform_matrix=transform,
        peak_threshold=0.8,
    )

    assert len(detections) >= 1
    det = detections[0]
    assert det.target_class == "aircraft_inflight"
    assert det.sensor == "S2_L1C"
    assert det.source_tile == "S2A_TEST_TILE"
    assert det.acquisition_time == acq_time
    assert det.geom_wgs84.is_valid
    assert det.confidence > 0.0
    assert det.length_m is not None
    assert det.heading_deg is not None
    assert 0.0 <= det.heading_deg <= 360.0


def test_parallax_detector_rejection_low_speed():
    scene = generate_synthetic_parallax_scene(
        height=80,
        width=80,
        triplets=[
            ((30, 30), (30, 31), (30, 32)),
        ],
    )

    detector = ParallaxInflightDetector(
        pixel_spacing_m=10.0,
        inter_band_delay_seconds=0.52,
        min_velocity_kmh=300.0,
        max_velocity_kmh=1000.0,
    )

    detections = detector.detect(scene, peak_threshold=0.8)
    assert len(detections) == 0


def test_parallax_detector_rejection_non_colinear():
    scene = generate_synthetic_parallax_scene(
        height=80,
        width=80,
        triplets=[
            ((30, 30), (35, 30), (35, 35)),
        ],
    )

    detector = ParallaxInflightDetector(
        pixel_spacing_m=10.0,
        inter_band_delay_seconds=0.52,
        min_velocity_kmh=300.0,
        max_velocity_kmh=1000.0,
        triplet_colinearity_max_deg=15.0,
    )

    detections = detector.detect(scene, peak_threshold=0.8)
    assert len(detections) == 0
