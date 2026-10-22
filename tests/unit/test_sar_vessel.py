from datetime import datetime, timezone
import numpy as np
import pytest
from src.core.cfar import OSCFARDetector
from src.detectors.sar_vessel import SarVesselDetector
from tests.mocks.mock_data import generate_synthetic_sar_scene


def test_sar_vessel_detector_on_synthetic_scene():
    scene = generate_synthetic_sar_scene(
        height=80,
        width=80,
        targets=[
            (25, 25, 5, 2, 500.0),
            (55, 55, 2, 6, 600.0),
        ],
    )

    cfar = OSCFARDetector(guard_m=20.0, train_m=60.0, pixel_spacing_m=10.0, pfa=1e-4)
    detector = SarVesselDetector(
        weights_config_path="configs/model_registry.yaml",
        cfar_detector=cfar,
        pixel_spacing_m=10.0,
    )

    transform = (0.0001, 0.0, 108.0, 0.0, -0.0001, 16.0)
    acq_time = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)

    detections = detector.detect(
        scene,
        sensor="S1_GRD",
        source_tile="TEST_SCENE_001",
        acquisition_time=acq_time,
        transform_matrix=transform,
    )

    assert len(detections) >= 2
    for det in detections:
        assert det.target_class == "vessel"
        assert det.sensor == "S1_GRD"
        assert det.source_tile == "TEST_SCENE_001"
        assert det.acquisition_time == acq_time
        assert det.geom_wgs84.is_valid
        assert det.confidence > 0.0
        assert det.length_m is not None
        assert det.heading_deg is not None
        assert 0.0 <= det.heading_deg <= 360.0


def test_sar_vessel_detector_heading_pca():
    scene = np.zeros((60, 60), dtype=np.float32)
    scene[22:27, 24:26] = 800.0

    cfar = OSCFARDetector(guard_m=40.0, train_m=80.0, pixel_spacing_m=10.0, pfa=1e-4)
    detector = SarVesselDetector(cfar_detector=cfar)

    detections = detector.detect(scene, transform_matrix=(0.0001, 0.0, 108.0, 0.0, -0.0001, 16.0))

    assert len(detections) == 1
    det = detections[0]
    assert det.heading_deg is not None
    assert (abs(det.heading_deg - 0.0) < 15.0) or (abs(det.heading_deg - 180.0) < 15.0) or (abs(det.heading_deg - 360.0) < 15.0)


def test_sar_vessel_detector_weights_fallback():
    detector = SarVesselDetector(weights_config_path="configs/model_registry.yaml")
    assert detector.mode == "heuristic_cfar"


@pytest.mark.weights
def test_sar_vessel_detector_weights_execution():
    detector = SarVesselDetector(weights_config_path="configs/model_registry.yaml")
    if detector.mode != "model_loaded":
        pytest.skip("Weights not downloaded locally.")
