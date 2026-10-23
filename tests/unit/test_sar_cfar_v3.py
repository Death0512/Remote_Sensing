from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pytest
from shapely.geometry import box
import yaml

from src.cli.main import cli_entrypoint
from src.detection.sar.local_cfar import LocalCfarDetector
from src.detection.sar.vessel_detector import SarVesselDetector
from src.domain.enums import SensorType, TargetClass
from src.domain.observation import ProcessedObservation
from src.domain.provenance import Provenance
from src.preprocessing.masking.land import LandMaskExtractor, extract_land_mask
from tests.mocks.mock_data import generate_synthetic_sar_scene


def _dummy_provenance(prov_id: str = "prov-sar-test") -> Provenance:
    return Provenance(
        provenance_id=prov_id,
        source_asset_id="asset-test-s1",
        acquisition_metadata_ref="s3://sentinel1/test.xml",
        pipeline_version="3.0.0",
        git_commit="aabbcc0",
        config_hash="cfg-hash-0",
        model_id=None,
        model_version=None,
        input_checksum="inchk0",
        output_checksum="outchk0",
        processing_run_id="run-test",
        processing_time_utc=datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc),
    )


def test_metric_window_parameterization_via_gsd():
    det_10m = LocalCfarDetector(
        guard_radius_m=40.0,
        background_radius_m=150.0,
        gsd_m=10.0,
    )
    assert det_10m.guard_radius_px == 4
    assert det_10m.background_radius_px == 15
    assert det_10m.guard_radius_m == 40.0
    assert det_10m.background_radius_m == 150.0

    det_20m = LocalCfarDetector(
        guard_radius_m=40.0,
        background_radius_m=150.0,
        gsd_m=20.0,
    )
    assert det_20m.guard_radius_px == 2
    assert det_20m.background_radius_px == 8

    det_5m = LocalCfarDetector(
        guard_radius_m=40.0,
        background_radius_m=150.0,
        gsd_m=5.0,
    )
    assert det_5m.guard_radius_px == 8
    assert det_5m.background_radius_px == 30

    det_px = LocalCfarDetector(
        guard_radius_px=3,
        background_radius_px=10,
        gsd_m=10.0,
    )
    assert det_px.guard_radius_px == 3
    assert det_px.background_radius_px == 10

    with pytest.raises(ValueError):
        LocalCfarDetector(gsd_m=-5.0)

    with pytest.raises(ValueError):
        LocalCfarDetector(guard_radius_px=-1)

    with pytest.raises(ValueError):
        LocalCfarDetector(guard_radius_px=10, background_radius_px=5)


def test_coastline_land_mask_prefiltering_avoids_clutter_inflation():
    rng = np.random.default_rng(101)
    ocean_clutter = rng.exponential(scale=5.0, size=(100, 100)).astype(np.float64)

    target_r, target_c = 50, 50
    ocean_clutter[target_r, target_c] = 75.0

    land_intensity = np.copy(ocean_clutter)
    land_intensity[45:55, 60:85] = 8000.0

    land_mask = np.zeros((100, 100), dtype=bool)
    land_mask[45:55, 60:85] = True

    extractor = LandMaskExtractor(buffer_pixels=2)
    buffered_mask = extractor.extract_land_mask(shape=(100, 100), land_mask=land_mask)
    assert np.sum(buffered_mask) > np.sum(land_mask)

    detector = LocalCfarDetector(
        guard_radius_px=2,
        background_radius_px=15,
        pfa=1e-3,
        cfar_type="CA",
    )

    det_unmasked, mu_unmasked, _ = detector.detect_mask(land_intensity, land_mask=None)
    assert not det_unmasked[target_r, target_c]
    assert mu_unmasked[target_r, target_c] > 100.0

    det_masked, mu_masked, _ = detector.detect_mask(land_intensity, land_mask=buffered_mask)
    assert det_masked[target_r, target_c]
    assert not np.any(det_masked[buffered_mask])
    assert mu_masked[target_r, target_c] < 10.0


def test_land_mask_extractor_polygon_rasterization():
    poly = box(20.0, 20.0, 60.0, 60.0)
    transform = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    extractor = LandMaskExtractor(buffer_pixels=1)
    mask = extractor.extract_land_mask(
        shape=(100, 100),
        polygons=[poly],
        transform=transform,
    )

    assert mask.shape == (100, 100)
    assert mask.dtype == bool
    assert mask[30, 30]
    assert not mask[5, 5]

    intensity = np.ones((100, 100), dtype=np.float32) * 50.0
    masked_img = extractor.apply_land_mask(intensity, mask, fill_value=0.0)
    assert masked_img[30, 30] == 0.0
    assert masked_img[5, 5] == 50.0


def test_ca_and_os_cfar_detection_on_synthetic_targets_with_known_snr():
    rng = np.random.default_rng(202)
    bg = rng.exponential(scale=10.0, size=(80, 80)).astype(np.float64)
    bg[40, 40] = 400.0

    ca_detector = LocalCfarDetector(
        guard_radius_m=30.0,
        background_radius_m=120.0,
        gsd_m=10.0,
        pfa=1e-3,
        cfar_type="CA",
    )
    ca_mask, ca_mu, ca_snr = ca_detector.detect_mask(bg)
    assert ca_mask[40, 40]
    assert 25.0 <= ca_snr[40, 40] <= 55.0

    ca_candidates = ca_detector.detect(bg)
    assert len(ca_candidates) >= 1
    target_cand = [
        c
        for c in ca_candidates
        if 38 <= dict(c.features)["centroid_row"] <= 42
        and 38 <= dict(c.features)["centroid_col"] <= 42
    ][0]
    assert dict(target_cand.features)["snr"] > 20.0
    assert target_cand.provenance.pipeline_version == "3.0.0"

    os_detector = LocalCfarDetector(
        guard_radius_m=30.0,
        background_radius_m=120.0,
        gsd_m=10.0,
        pfa=1e-3,
        cfar_type="OS",
        k_order_ratio=0.75,
    )
    os_mask, os_mu, os_snr = os_detector.detect_mask(bg)
    assert os_mask[40, 40]
    assert os_snr[40, 40] > 15.0

    os_candidates = os_detector.detect(bg)
    assert len(os_candidates) >= 1


def test_cfar_scale_invariance_ca_and_os():
    rng = np.random.default_rng(303)
    base_img = rng.exponential(scale=3.0, size=(60, 60)).astype(np.float64)
    base_img[30:33, 30:33] = 120.0

    ca_det = LocalCfarDetector(guard_radius_px=2, background_radius_px=8, pfa=1e-4, cfar_type="CA")
    ca_base, _, _ = ca_det.detect_mask(base_img)
    ca_scaled_up, _, _ = ca_det.detect_mask(base_img * 2500.0)
    ca_scaled_down, _, _ = ca_det.detect_mask(base_img * 0.0004)
    assert np.array_equal(ca_base, ca_scaled_up)
    assert np.array_equal(ca_base, ca_scaled_down)

    os_det = LocalCfarDetector(guard_radius_px=2, background_radius_px=6, pfa=1e-3, cfar_type="OS")
    os_base, _, _ = os_det.detect_mask(base_img)
    os_scaled_up, _, _ = os_det.detect_mask(base_img * 3200.0)
    os_scaled_down, _, _ = os_det.detect_mask(base_img * 0.005)
    assert np.array_equal(os_base, os_scaled_up)
    assert np.array_equal(os_base, os_scaled_down)


def test_morphological_filtering_vessel_detector():
    img = np.ones((120, 120), dtype=np.float64) * 2.0
    img[20:28, 20:23] = 200.0
    img[60:61, 60:61] = 200.0
    img[90:93, 90:93] = 200.0

    cfar = LocalCfarDetector(guard_radius_px=3, background_radius_px=10, pfa=1e-3)
    detector = SarVesselDetector(
        cfar_detector=cfar,
        min_length_m=20.0,
        max_length_m=150.0,
        min_width_m=2.0,
        max_width_m=40.0,
        min_aspect_ratio=1.5,
        max_aspect_ratio=8.0,
        pixel_spacing_m=10.0,
    )

    detections = detector.detect(intensity=img)
    assert len(detections) == 1
    det = detections[0]
    assert det.target_class == TargetClass.VESSEL
    assert det.sensor == SensorType.S1_SAR
    assert det.provenance.pipeline_version == "3.0.0"

    attrs = dict(det.attributes)
    assert 20.0 <= attrs["length_m"] <= 150.0
    assert 1.5 <= attrs["aspect_ratio"] <= 8.0


def test_cli_scan_sar_integration_with_land_mask(tmp_path: Path, capsys):
    scene = generate_synthetic_sar_scene(
        height=80,
        width=80,
        targets=[(30, 30, 4, 2, 400.0)],
        seed=404,
    )
    img_path = tmp_path / "test_sar.npy"
    np.save(str(img_path), scene)

    land_mask = np.zeros((80, 80), dtype=bool)
    land_mask[60:80, 60:80] = True
    mask_path = tmp_path / "land_mask.npy"
    np.save(str(mask_path), land_mask)

    out_dir = tmp_path / "sar_scan_output"
    exit_code = cli_entrypoint([
        "scan",
        "--image", str(img_path),
        "--sensor", "S1_SAR",
        "--land-mask", str(mask_path),
        "--output-dir", str(out_dir),
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Scan complete" in captured.out

    alerts_path = out_dir / "alerts.geojson"
    assert alerts_path.exists()
    with open(alerts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) >= 1
    feat = data["features"][0]
    assert feat["properties"]["sensor"] == "S1_SAR"
    assert feat["properties"]["target_class"] == "TargetClass.VESSEL" or "VESSEL" in feat["properties"]["target_class"]


def test_unified_sar_cfar_config_schema():
    cfg_path = Path("configs/detection/sar_cfar.yaml")
    assert cfg_path.exists()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["schema_version"] == "sar_cfar.v1"
    assert "ca_cfar" in cfg
    assert "os_cfar" in cfg

    ca_cfg = cfg["ca_cfar"]
    ca_detector = LocalCfarDetector(
        guard_radius_m=ca_cfg["guard_radius_m"],
        background_radius_m=ca_cfg["background_radius_m"],
        gsd_m=ca_cfg["gsd_m"],
        pfa=ca_cfg["pfa"],
        cfar_type=ca_cfg["cfar_type"],
        min_training_cells=ca_cfg["min_training_cells"],
    )
    assert ca_detector.cfar_type == "CA"

    os_cfg = cfg["os_cfar"]
    os_detector = LocalCfarDetector(
        guard_radius_m=os_cfg["guard_radius_m"],
        background_radius_m=os_cfg["background_radius_m"],
        gsd_m=os_cfg["gsd_m"],
        pfa=os_cfg["pfa"],
        cfar_type=os_cfg["cfar_type"],
        k_order_ratio=os_cfg["k_order_ratio"],
        min_training_cells=os_cfg["min_training_cells"],
    )
    assert os_detector.cfar_type == "OS"

    vessel_det = SarVesselDetector(
        cfar_detector=ca_detector,
        min_length_m=ca_cfg["min_length_m"],
        max_length_m=ca_cfg["max_length_m"],
        min_width_m=ca_cfg["min_width_m"],
        max_width_m=ca_cfg["max_width_m"],
        min_aspect_ratio=ca_cfg["min_aspect_ratio"],
        max_aspect_ratio=ca_cfg["max_aspect_ratio"],
        pixel_spacing_m=ca_cfg["gsd_m"],
    )
    assert vessel_det.min_length_m == 10.0
    assert vessel_det.max_length_m == 450.0
