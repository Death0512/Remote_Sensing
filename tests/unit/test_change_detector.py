from datetime import datetime, timezone
import pytest
from src.detectors.change_detector import DualStreamChangeDetector
from tests.mocks.mock_data import generate_synthetic_bitemporal_pair


def test_change_detector_bitemporal():
    t1, t2 = generate_synthetic_bitemporal_pair(
        height=100,
        width=100,
        new_structures=[
            (40, 40, 16, 16, 0.9),
            (75, 75, 4, 4, 0.85),
        ],
    )

    detector = DualStreamChangeDetector(
        weights_config_path="configs/model_registry.yaml",
        pixel_spacing_m=10.0,
        diff_threshold=0.3,
        min_area_px=15,
        structure_area_threshold_m2=2500.0,
    )

    transform = (0.0001, 0.0, 108.0, 0.0, -0.0001, 16.0)
    acq_time = datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc)

    detections = detector.detect(
        (t1, t2),
        sensor="S1_RTC",
        source_tile="BITEMPORAL_001",
        acquisition_time=acq_time,
        transform_matrix=transform,
    )

    assert len(detections) >= 2
    classes = {det.target_class for det in detections}
    assert "new_structure" in classes
    assert "unknown_structure" in classes
    for det in detections:
        assert det.sensor == "S1_RTC"
        assert det.source_tile == "BITEMPORAL_001"
        assert det.acquisition_time == acq_time
        assert det.geom_wgs84.is_valid
        assert det.confidence > 0.0
        assert det.length_m is not None
        assert det.length_m > 0.0


def test_change_detector_area_thresholding():
    t1, t2 = generate_synthetic_bitemporal_pair(
        height=100,
        width=100,
        new_structures=[
            (10, 10, 20, 20, 0.9),
            (60, 60, 4, 4, 0.9),
        ],
    )

    detector = DualStreamChangeDetector(
        weights_config_path="configs/model_registry.yaml",
        pixel_spacing_m=10.0,
        diff_threshold=0.3,
        min_area_px=10,
        structure_area_threshold_m2=2500.0,
    )

    detections = detector.detect((t1, t2))
    assert len(detections) == 2

    large_det = next(d for d in detections if d.geom_px[2] - d.geom_px[0] >= 15)
    small_det = next(d for d in detections if d.geom_px[2] - d.geom_px[0] < 15)

    assert large_det.target_class == "new_structure"
    assert small_det.target_class == "unknown_structure"


def test_change_detector_cosine_suppression():
    t1, t2 = generate_synthetic_bitemporal_pair(
        height=60,
        width=60,
        new_structures=[
            (20, 20, 10, 10, 0.9),
        ],
    )

    detector = DualStreamChangeDetector(
        weights_config_path="configs/model_registry.yaml",
        diff_threshold=0.3,
        cosine_threshold=0.98,
        min_area_px=10,
    )

    detections = detector.detect((t1, t2), apply_cosine_filter=True)
    assert len(detections) >= 1


@pytest.mark.weights
def test_change_detector_weights():
    detector = DualStreamChangeDetector(weights_config_path="configs/model_registry.yaml")
    if detector.mode != "model_loaded":
        pytest.skip("DS-UNet weights not downloaded locally.")
