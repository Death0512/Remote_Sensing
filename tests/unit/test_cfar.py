import numpy as np
from src.core.cfar import OSCFARDetector


def test_cfar_detection_with_synthetic_target():
    np.random.seed(42)
    background = np.random.exponential(scale=10.0, size=(50, 50))
    background[25, 25] = 500.0

    detector = OSCFARDetector(guard_m=20.0, train_m=60.0, pixel_spacing_m=10.0, pfa=1e-3)
    detections = detector.detect(background)

    assert detections[25, 25] is np.True_ or detections[25, 25] == 1
    assert detections[:10, :10].sum() == 0


def test_cfar_land_mask_suppression():
    background = np.ones((50, 50)) * 10.0
    background[25, 25] = 500.0
    land_mask = np.zeros((50, 50), dtype=bool)
    land_mask[25, 25] = True

    detector = OSCFARDetector(guard_m=20.0, train_m=60.0, pixel_spacing_m=10.0, pfa=1e-3)
    detections = detector.detect(background, land_mask=land_mask)

    assert detections[25, 25] is np.False_ or detections[25, 25] == 0
