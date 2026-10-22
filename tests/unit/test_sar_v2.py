from datetime import datetime, timezone
import numpy as np
import pytest
from shapely.geometry import Polygon

from src.detection.sar.local_cfar import LocalCfarDetector
from src.detection.sar.vessel_detector import SarVesselDetector
from src.domain.confidence import ConfidenceVector
from src.domain.detection import Detection
from src.domain.enums import (
    CalibrationState,
    Capability,
    QualityTier,
    RadiometricState,
    SensorType,
    TargetClass,
    Zone,
)
from src.domain.observation import Observation
from src.domain.provenance import Provenance
from src.preprocessing.geometry.geofence import GeofenceManager
from src.preprocessing.geometry.tiling import (
    MetricTiler,
    generate_metric_tiling_windows,
    point_deduplicate_metric,
)
from src.preprocessing.sar.preprocessor import SarPreprocessor


def _dummy_provenance(prov_id: str = "prov-root") -> Provenance:
    return Provenance(
        provenance_id=prov_id,
        source_asset_id="asset-sar-001",
        acquisition_metadata_ref="s3://sentinel1/meta.xml",
        pipeline_version="2.0.0",
        git_commit="fedcba9",
        config_hash="cfg-sar-01",
        model_id=None,
        model_version=None,
        input_checksum="input-chk-1",
        output_checksum="output-chk-1",
        processing_run_id="run-init",
        processing_time_utc=datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc),
    )


def test_cfar_is_purely_local_invariant_to_global_scaling() -> None:
    rng = np.random.default_rng(42)
    clutter = rng.exponential(scale=2.0, size=(120, 120)).astype(np.float64)
    clutter[30:33, 30:33] = 80.0
    clutter[80:83, 80:83] = 95.0

    detector = LocalCfarDetector(
        guard_radius_px=2,
        background_radius_px=7,
        pfa=1e-4,
        cfar_type="CA",
    )
    assert detector.capability == Capability.IMPLEMENTED

    mask_base, _, _ = detector.detect_mask(clutter)
    mask_scaled_up, _, _ = detector.detect_mask(clutter * 1234.5)
    mask_scaled_down, _, _ = detector.detect_mask(clutter * 0.005)

    assert np.array_equal(mask_base, mask_scaled_up)
    assert np.array_equal(mask_base, mask_scaled_down)
    assert np.sum(mask_base) > 0

    cands_base = detector.detect(clutter)
    cands_scaled = detector.detect(clutter * 1234.5)
    assert len(cands_base) == len(cands_scaled)
    assert len(cands_base) >= 2


def test_landmask_strictly_excludes_coastal_pixels() -> None:
    intensity = np.ones((100, 100), dtype=np.float64)
    intensity[:35, :35] = 200.0
    intensity[10, 10] = 5000.0
    intensity[38, 38] = 25.0

    land_mask = np.zeros((100, 100), dtype=bool)
    land_mask[:35, :35] = True

    detector = LocalCfarDetector(
        guard_radius_px=2,
        background_radius_px=8,
        pfa=1e-3,
        min_training_cells=8,
    )

    mask_with_land, _, _ = detector.detect_mask(intensity, land_mask=land_mask)
    assert not np.any(mask_with_land[land_mask])
    assert mask_with_land[38, 38]

    mask_without_land, _, _ = detector.detect_mask(intensity, land_mask=None)
    assert mask_without_land[10, 10]
    assert not mask_without_land[38, 38]


def test_zone_classification_outside_aoi() -> None:
    manager = GeofenceManager()
    assert manager.capability == Capability.IMPLEMENTED

    z_center = manager.classify_zone(16.0, 108.0)
    assert z_center == Zone.ZONE_1

    z_mid = manager.classify_zone(16.0, 117.0)
    assert z_mid == Zone.ZONE_2

    z_outer = manager.classify_zone(16.0, 126.0)
    assert z_outer == Zone.ZONE_3

    z_far_north = manager.classify_zone(45.0, 108.0)
    assert z_far_north == Zone.OUTSIDE_AOI

    z_far_equator = manager.classify_zone(0.0, 0.0)
    assert z_far_equator == Zone.OUTSIDE_AOI


def test_gsd_aware_metric_tiling() -> None:
    tiler_10m = MetricTiler(
        metric_window_m=1000.0,
        metric_overlap_m=200.0,
        gsd_m=10.0,
    )
    assert tiler_10m.capability == Capability.IMPLEMENTED
    assert tiler_10m.window_px == 100
    assert tiler_10m.overlap_px == 20
    assert tiler_10m.step_px == 80

    tiler_5m = MetricTiler(
        metric_window_m=1000.0,
        metric_overlap_m=200.0,
        gsd_m=5.0,
    )
    assert tiler_5m.window_px == 200
    assert tiler_5m.overlap_px == 40
    assert tiler_5m.step_px == 160

    windows = list(
        generate_metric_tiling_windows(
            image_width=250,
            image_height=250,
            metric_window_m=1000.0,
            metric_overlap_m=200.0,
            gsd_m=10.0,
        )
    )
    assert len(windows) > 1
    for win in windows:
        assert win.width <= 100
        assert win.height <= 100
        assert win.col_slice == slice(win.col_off, win.col_off + win.width)
        assert win.row_slice == slice(win.row_off, win.row_off + win.height)

    pts = [(10.0, 10.0, 0.9), (12.0, 12.0, 0.5), (30.0, 30.0, 0.8)]
    deduped_10m = point_deduplicate_metric(pts, metric_dedup_m=50.0, gsd_m=10.0)
    assert len(deduped_10m) == 2
    deduped_1m = point_deduplicate_metric(pts, metric_dedup_m=50.0, gsd_m=1.0)
    assert len(deduped_1m) == 1

    with pytest.raises(ValueError):
        MetricTiler(metric_window_m=500.0, metric_overlap_m=600.0, gsd_m=10.0)
    with pytest.raises(ValueError):
        MetricTiler(metric_window_m=500.0, metric_overlap_m=100.0, gsd_m=-5.0)


def test_sar_vessel_detection_contract() -> None:
    poly = Polygon([(107.0, 15.0), (109.0, 15.0), (109.0, 17.0), (107.0, 17.0)])
    prov = _dummy_provenance()

    calibrated_obs = Observation(
        observation_id="obs-s1-cal",
        source="CDSE",
        platform="Sentinel-1A",
        sensor=SensorType.S1_SAR,
        product="GRD",
        processing_level="LEVEL1",
        acquisition_time_utc=datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc),
        crs="EPSG:4326",
        ground_sample_distance_m=10.0,
        bounds_wgs84=poly,
        asset_uri="s3://sentinel1/grd.safe",
        metadata_uri="s3://sentinel1/calibration.xml",
        quality_flags=("CALIBRATED_SIGMA0",),
        provenance=prov,
    )

    uncalibrated_obs = Observation(
        observation_id="obs-s1-raw",
        source="CDSE",
        platform="Sentinel-1A",
        sensor=SensorType.S1_SAR,
        product="RAW",
        processing_level="LEVEL0",
        acquisition_time_utc=datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc),
        crs="EPSG:4326",
        ground_sample_distance_m=10.0,
        bounds_wgs84=poly,
        asset_uri="s3://sentinel1/raw.safe",
        metadata_uri="",
        quality_flags=("UNCALIBRATED",),
        provenance=prov,
    )

    preprocessor = SarPreprocessor(buffer_pixels=2)
    assert preprocessor.capability == Capability.IMPLEMENTED

    proc_cal = preprocessor.process(calibrated_obs)
    assert proc_cal.quality_tier == QualityTier.DATA_OK
    assert proc_cal.radiometric_state == RadiometricState.CALIBRATED_SIGMA0

    proc_uncal = preprocessor.process(uncalibrated_obs)
    assert proc_uncal.quality_tier == QualityTier.RADIOMETRIC_UNCERTAIN
    assert proc_uncal.radiometric_state == RadiometricState.RAW

    detector = SarVesselDetector(
        cfar_detector=LocalCfarDetector(
            guard_radius_px=2,
            background_radius_px=6,
            pfa=1e-4,
        ),
        min_length_m=10.0,
        max_length_m=450.0,
        pixel_spacing_m=10.0,
    )
    assert detector.capability == Capability.IMPLEMENTED

    intensity = np.ones((100, 100), dtype=np.float64)
    intensity[25:30, 25:27] = 80.0
    intensity[70:71, 70:71] = 75.0

    detections = detector.detect(
        processed_observation=proc_cal,
        intensity=intensity,
        pixel_spacing_m=10.0,
    )

    assert isinstance(detections, tuple)
    assert len(detections) >= 1
    for det in detections:
        assert isinstance(det, Detection)
        assert det.target_class == TargetClass.VESSEL
        assert det.sensor == SensorType.S1_SAR
        assert isinstance(det.confidence, ConfidenceVector)
        assert det.confidence.calibrated_probability is None
        assert det.confidence.calibration_state == CalibrationState.NOT_CALIBRATED
        assert det.confidence.raw_score > 0.0
        assert det.provenance.parent_provenance_id is not None

        attr_dict = dict(det.attributes)
        assert "length_m" in attr_dict
        assert "width_m" in attr_dict
        assert "orientation_deg" in attr_dict
        assert "aspect_ratio" in attr_dict
        assert 10.0 <= attr_dict["length_m"] <= 450.0

    large_intensity = np.ones((100, 100), dtype=np.float64)
    large_intensity[10:70, 10:12] = 100.0
    large_detections = detector.detect(
        processed_observation=proc_cal,
        intensity=large_intensity,
        pixel_spacing_m=10.0,
    )
    for det in large_detections:
        attr_dict = dict(det.attributes)
        assert attr_dict["length_m"] <= 450.0
