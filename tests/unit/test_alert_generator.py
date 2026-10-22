from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytest
from shapely.geometry import Point

from src.core.contracts import Detection
from src.engine.alert_generator import AlertGenerator


def _create_detection(geom_px=(50.0, 50.0, 70.0, 70.0)) -> Detection:
    return Detection(
        det_id="DET_TEST_001",
        sensor="S1_GRD",
        geom_px=geom_px,
        geom_wgs84=Point(108.2, 16.1),
        crs="EPSG:4326",
        confidence=0.92,
        length_m=45.0,
        heading_deg=120.0,
        target_class="vessel",
        source_tile="test_tile",
        acquisition_time=datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc),
    )


def test_alert_generator_levels_and_zones(tmp_path: Path):
    generator = AlertGenerator(output_dir=tmp_path)
    det = _create_detection()
    img = np.ones((100, 100), dtype=np.float32)

    alert_z1 = generator.generate_alert(det, zone=1, score_s=85.0, source_image=img)
    assert alert_z1.level == "CRITICAL"
    assert alert_z1.zone == 1
    assert alert_z1.audit_status == "PENDING"
    assert Path(alert_z1.chip_path).exists()

    alert_z2 = generator.generate_alert(det, zone=2, score_s=60.0, source_image=img)
    assert alert_z2.level == "HIGH"
    assert alert_z2.zone == 2

    alert_z3 = generator.generate_alert(det, zone=3, score_s=30.0, source_image=img)
    assert alert_z3.level == "STRATEGIC"
    assert alert_z3.zone == 3

    with pytest.raises(ValueError):
        generator.generate_alert(det, zone=0, score_s=10.0, source_image=img)


def test_alert_geojson_attributes(tmp_path: Path):
    generator = AlertGenerator(output_dir=tmp_path)
    det = _create_detection()
    img = np.full((120, 120), 10.0, dtype=np.float32)

    alert = generator.generate_alert(det, zone=1, score_s=90.0, source_image=img, is_dark=True)
    feature = alert.geojson_feature

    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Point"
    assert tuple(feature["geometry"]["coordinates"]) == (108.2, 16.1)

    props = feature["properties"]
    assert props["target_class"] == "vessel"
    assert props["confidence"] == 0.92
    assert props["sensor"] == "S1_GRD"
    assert props["score_s"] == 90.0
    assert props["zone"] == 1
    assert props["level"] == "CRITICAL"
    assert props["is_dark"] is True
    assert "acquisition_time" in props


def test_chip_extraction_and_png_integrity(tmp_path: Path):
    generator = AlertGenerator(output_dir=tmp_path, chip_size=256)
    det = _create_detection(geom_px=(10.0, 10.0, 20.0, 20.0))
    img = np.random.default_rng(42).integers(0, 255, size=(64, 64), dtype=np.uint8)

    alert = generator.generate_alert(det, zone=1, score_s=75.0, source_image=img)
    chip_file = Path(alert.chip_path)

    assert chip_file.exists()
    content = chip_file.read_bytes()
    assert content.startswith(b"\x89PNG\r\n\x1a\n")


def test_chip_extraction_rgb_and_padding_modes(tmp_path: Path):
    generator = AlertGenerator(output_dir=tmp_path, chip_size=256, padding_mode="constant")
    det_out = _create_detection(geom_px=(1000.0, 1000.0, 1010.0, 1010.0))

    img_rgb = np.zeros((3, 50, 50), dtype=np.float32)
    chip = generator.extract_chip(img_rgb, det_out.geom_px)
    assert chip.shape == (3, 256, 256)
    assert np.all(chip == 0)

    img_hwc = np.ones((50, 50, 3), dtype=np.uint8) * 128
    chip_hwc = generator.extract_chip(img_hwc, (20.0, 20.0, 30.0, 30.0))
    assert chip_hwc.shape == (256, 256, 3)


def test_update_audit_status(tmp_path: Path):
    generator = AlertGenerator(output_dir=tmp_path)
    det = _create_detection()
    img = np.zeros((50, 50), dtype=np.float32)

    alert = generator.generate_alert(det, zone=2, score_s=50.0, source_image=img)
    assert alert.audit_status == "PENDING"

    confirmed = generator.update_audit_status(alert, "CONFIRMED")
    assert confirmed.audit_status == "CONFIRMED"
    assert confirmed.alert_id == alert.alert_id

    dismissed = generator.update_audit_status(alert, "DISMISSED")
    assert dismissed.audit_status == "DISMISSED"

    with pytest.raises(ValueError):
        generator.update_audit_status(alert, "INVALID_STATUS")
