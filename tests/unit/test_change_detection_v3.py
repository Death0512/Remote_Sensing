from datetime import datetime, timezone
import numpy as np
import pytest
import scipy.ndimage
from shapely.geometry import Polygon

from src.detection.change.change_detector import (
    BitemporalChangeDetector,
    CalibratedChangeDetector,
    DualStreamChangeDetector,
    RegistrationFailureReason,
    SkippedLowQuality,
)
from src.domain.confidence import ConfidenceVector
from src.domain.detection import Detection
from src.domain.enums import (
    CalibrationState,
    Capability,
    QualityTier,
    SensorType,
    TargetClass,
)
from src.domain.geometry import AxisAlignedBoundingBox
from src.domain.observation import Observation
from src.domain.provenance import Provenance
from src.preprocessing.registration.bitemporal import (
    BitemporalRegistration,
    RegistrationResult,
)


def _make_provenance(prov_id: str = "prov-test-v3") -> Provenance:
    return Provenance(
        provenance_id=prov_id,
        source_asset_id="asset-v3-001",
        acquisition_metadata_ref="s3://sentinel/metadata_v3.xml",
        pipeline_version="3.0.0",
        git_commit="git6f84f43",
        config_hash="cfg-v3-sha",
        model_id=None,
        model_version=None,
        input_checksum="inchk-v3",
        output_checksum="outchk-v3",
        processing_run_id="run-v3-001",
        processing_time_utc=datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc),
    )


def _make_observation(
    obs_id: str = "obs-001",
    sensor: SensorType = SensorType.S2_MSI,
    product: str = "S2MSI1C",
    crs: str = "EPSG:4326",
    gsd_m: float = 10.0,
    bounds: Polygon = Polygon([(107.0, 15.0), (109.0, 15.0), (109.0, 17.0), (107.0, 17.0)]),
) -> Observation:
    return Observation(
        observation_id=obs_id,
        source="CDSE",
        platform="Sentinel-2A" if sensor == SensorType.S2_MSI else "Sentinel-1A",
        sensor=sensor,
        product=product,
        processing_level="L1C",
        acquisition_time_utc=datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc),
        crs=crs,
        ground_sample_distance_m=gsd_m,
        bounds_wgs84=bounds,
        asset_uri=f"s3://sentinel/{obs_id}.safe",
        metadata_uri=f"s3://sentinel/{obs_id}_mtd.xml",
        quality_flags=("DATA_OK",),
        provenance=_make_provenance(f"prov-{obs_id}"),
    )


def test_registration_separation_displacement_and_post_warp_residual() -> None:
    reg = BitemporalRegistration(max_residual_rmse_px=0.75)
    obs_a = _make_observation("obs-a")
    obs_b = _make_observation("obs-b")

    np.random.seed(42)
    base = np.random.normal(0.5, 0.1, (100, 100)).astype(np.float32)
    base[30:60, 30:60] += 0.8
    base[10:25, 70:85] += 0.5
    base = np.clip(base, 0.0, 2.0)

    shift_y, shift_x = 2.0, 1.5
    arr_b = scipy.ndimage.shift(base, (shift_y, shift_x), order=1, mode="nearest")

    result = reg.align(obs_a, obs_b, base, arr_b)

    assert isinstance(result, RegistrationResult)
    assert result.success is True
    assert result.quality_tier == QualityTier.DATA_OK

    est_x, est_y = result.estimated_displacement_px
    disp_magnitude = np.hypot(est_x, est_y)
    assert disp_magnitude > 1.5
    assert abs(est_x - shift_x) < 0.5
    assert abs(est_y - shift_y) < 0.5

    assert result.residual_rmse_px < 0.5
    assert result.residual_rmse_px < disp_magnitude
    assert result.residual_median_px >= 0.0
    assert result.residual_p95_px >= result.residual_median_px
    assert result.transform_matrix is not None


def test_registration_aborts_change_detection_when_residual_exceeds_threshold() -> None:
    reg = BitemporalRegistration(max_residual_rmse_px=0.3)
    obs_a = _make_observation("obs-a")
    obs_b = _make_observation("obs-b")

    np.random.seed(42)
    arr_a = np.zeros((120, 120), dtype=np.float32)
    arr_a[20:50, 20:50] = 1.0
    arr_a[70:100, 70:100] = 0.8

    arr_b = arr_a.copy()
    arr_b[:60, :] = scipy.ndimage.shift(arr_b[:60, :], (0.0, 3.0), order=1, mode="nearest")
    arr_b[60:, :] = scipy.ndimage.shift(arr_b[60:, :], (0.0, -3.0), order=1, mode="nearest")

    result = reg.align(obs_a, obs_b, arr_a, arr_b)

    assert result.success is False
    assert result.quality_tier == QualityTier.GEOREGISTRATION_UNCERTAIN
    assert result.residual_rmse_px > 0.3
    assert result.failure_reason is not None

    detector = BitemporalChangeDetector()
    detections = detector.detect(result)

    assert isinstance(detections, SkippedLowQuality)
    assert len(detections) == 0
    assert detections.reason == "REGISTRATION_FAILED"


def test_sensor_and_crs_compatibility_enforced() -> None:
    reg = BitemporalRegistration()
    arr = np.zeros((50, 50), dtype=np.float32)

    obs_s1 = _make_observation("obs-s1", sensor=SensorType.S1_SAR, product="S1_GRD")
    obs_s2 = _make_observation("obs-s2", sensor=SensorType.S2_MSI, product="S2MSI1C")
    with pytest.raises(ValueError, match="Incompatible observation sensors"):
        reg.align(obs_s1, obs_s2, arr, arr)

    obs_crs1 = _make_observation("obs-c1", crs="EPSG:4326")
    obs_crs2 = _make_observation("obs-c2", crs="EPSG:32632")
    with pytest.raises(ValueError, match="Incompatible observation CRS"):
        reg.align(obs_crs1, obs_crs2, arr, arr)

    obs_prod1 = _make_observation("obs-p1", product="S2MSI1C")
    obs_prod2 = _make_observation("obs-p2", product="S2MSI2A")
    with pytest.raises(ValueError, match="Incompatible observation products"):
        reg.align(obs_prod1, obs_prod2, arr, arr)


def test_corrupted_metadata_handling() -> None:
    reg = BitemporalRegistration()
    arr = np.zeros((50, 50), dtype=np.float32)
    obs_valid = _make_observation("obs-valid")
    obs_corrupted = _make_observation("obs-corrupt", bounds=Polygon())

    result = reg.align(obs_valid, obs_corrupted, arr, arr)
    assert result.success is False
    assert result.quality_tier == QualityTier.CORRUPTED_METADATA

    detector = BitemporalChangeDetector()
    detections = detector.detect(result)
    assert isinstance(detections, SkippedLowQuality)
    assert detections.reason == "CORRUPTED_METADATA"


def test_calibrated_detection_adapts_to_clean_background_detecting_subtle_change() -> None:
    reg = BitemporalRegistration(max_residual_rmse_px=0.75)
    obs_a = _make_observation("obs-clean-a")
    obs_b = _make_observation("obs-clean-b")

    np.random.seed(123)
    arr_a = np.full((100, 100), 0.3, dtype=np.float32)
    arr_a += np.random.normal(0.0, 0.003, (100, 100)).astype(np.float32)

    arr_b = arr_a.copy()
    arr_b[40:55, 40:55] += 0.12

    result = reg.align(obs_a, obs_b, arr_a, arr_b)
    assert result.success is True

    detector = BitemporalChangeDetector(
        sigma_multiplier=3.0,
        min_area_px=10,
        morph_kernel_size=1,
    )
    detections = detector.detect(result)

    assert len(detections) >= 1
    det = detections[0]
    th = dict(det.attributes)["calibrated_threshold"]
    assert th < 0.1
    assert det.target_class == TargetClass.INFRASTRUCTURE_CHANGE
    assert det.confidence.raw_score > 0.0


def test_calibrated_detection_suppresses_high_variance_noisy_background() -> None:
    reg = BitemporalRegistration(max_residual_rmse_px=0.75)
    obs_a = _make_observation("obs-noisy-a")
    obs_b = _make_observation("obs-noisy-b")

    np.random.seed(456)
    base = np.random.normal(0.5, 0.08, (100, 100)).astype(np.float32)
    arr_a = base + np.random.normal(0.0, 0.02, (100, 100)).astype(np.float32)
    arr_b = base + np.random.normal(0.0, 0.02, (100, 100)).astype(np.float32)

    result = reg.align(obs_a, obs_b, arr_a, arr_b)
    assert result.success is True

    detector = BitemporalChangeDetector(
        sigma_multiplier=3.5,
        min_area_px=15,
        morph_kernel_size=3,
    )
    detections = detector.detect(result)
    assert len(detections) == 0


def test_v3_domain_contracts_and_aliases() -> None:
    reg = BitemporalRegistration()
    obs_a = _make_observation("obs-a")
    obs_b = _make_observation("obs-b")

    arr_a = np.full((80, 80), 0.2, dtype=np.float32)
    arr_b = arr_a.copy()
    arr_b[25:45, 25:45] = 0.9

    result = reg.align(obs_a, obs_b, arr_a, arr_b)
    assert result.success is True

    assert DualStreamChangeDetector is BitemporalChangeDetector
    assert CalibratedChangeDetector is BitemporalChangeDetector

    detector = DualStreamChangeDetector(min_area_px=10, morph_kernel_size=1)
    detections = detector.detect(result)

    assert len(detections) == 1
    det = detections[0]

    assert isinstance(det, Detection)
    assert det.target_class == TargetClass.INFRASTRUCTURE_CHANGE
    assert isinstance(det.confidence, ConfidenceVector)
    assert det.confidence.calibration_state == CalibrationState.NOT_CALIBRATED
    assert det.confidence.quality_score == 1.0
    assert det.bounding_box is not None
    assert isinstance(det.bounding_box, AxisAlignedBoundingBox)
    assert det.geom_wgs84 is not None
    assert not det.geom_wgs84.is_empty
    assert det.provenance is not None
    assert det.provenance.pipeline_version == "3.0.0"

    attrs = dict(det.attributes)
    assert "calibrated_threshold" in attrs
    assert "area_m2" in attrs
    assert "length_m" in attrs
    assert "width_m" in attrs
    assert "residual_px" in attrs
    assert "pixel_geometry_row" in attrs
    assert "pixel_geometry_col" in attrs


def test_direct_array_pair_detection() -> None:
    arr_a = np.full((60, 60), 0.2, dtype=np.float32)
    arr_b = arr_a.copy()
    arr_b[20:35, 20:35] = 0.85

    detector = BitemporalChangeDetector(min_area_px=5, morph_kernel_size=1)
    detections = detector.detect((arr_a, arr_b))

    assert len(detections) >= 1
    det = detections[0]
    assert det.target_class == TargetClass.INFRASTRUCTURE_CHANGE
    assert det.confidence.raw_score > 0.0
