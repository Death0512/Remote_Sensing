from datetime import datetime, timezone
import pytest
from src.detectors.optical_aircraft import OpticalAircraftDetector
from tests.mocks.mock_data import generate_synthetic_optical_airport


def test_optical_aircraft_detector_heuristic():
    scene = generate_synthetic_optical_airport(
        height=100,
        width=100,
        aircraft_positions=[
            (25, 25, 10, 6),
            (60, 60, 12, 6),
        ],
    )

    detector = OpticalAircraftDetector(
        weights_config_path="configs/model_registry.yaml",
        pixel_spacing_m=0.5,
        intensity_threshold=0.6,
        min_area_px=10,
    )

    transform = (0.00005, 0.0, 108.0, 0.0, -0.00005, 16.0)
    acq_time = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)

    detections = detector.detect(
        scene,
        sensor="VHR",
        source_tile="TEST_AIRPORT_01",
        acquisition_time=acq_time,
        transform_matrix=transform,
    )

    assert len(detections) >= 2
    for det in detections:
        assert det.target_class == "aircraft_parked"
        assert det.sensor == "VHR"
        assert det.source_tile == "TEST_AIRPORT_01"
        assert det.acquisition_time == acq_time
        assert det.geom_wgs84.is_valid
        assert det.confidence > 0.0
        assert det.length_m is not None
        assert det.length_m > 0.0
        assert det.heading_deg is not None
        assert 0.0 <= det.heading_deg <= 360.0


def test_optical_aircraft_detector_obb_geometry():
    scene = generate_synthetic_optical_airport(
        height=80,
        width=80,
        aircraft_positions=[(30, 30, 14, 8)],
    )

    detector = OpticalAircraftDetector(
        weights_config_path="configs/model_registry.yaml",
        pixel_spacing_m=1.0,
        intensity_threshold=0.5,
        min_area_px=15,
    )

    transform = (0.0001, 0.0, 108.0, 0.0, -0.0001, 16.0)
    detections = detector.detect(scene, transform_matrix=transform)

    assert len(detections) == 1
    det = detections[0]
    poly = det.geom_wgs84
    assert poly.geom_type == "Polygon"
    assert len(poly.exterior.coords) == 5


@pytest.mark.weights
def test_optical_aircraft_detector_weights():
    detector = OpticalAircraftDetector(weights_config_path="configs/model_registry.yaml")
    if detector.mode != "model_loaded":
        pytest.skip("YOLOv8-OBB weights not downloaded locally.")
