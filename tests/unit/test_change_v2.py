from datetime import datetime, timezone
import numpy as np
import pytest
import scipy.ndimage
from shapely.geometry import Polygon

from src.detection.change.change_detector import (
    BitemporalChangeDetector,
    SkippedLowQuality,
)
from src.domain.enums import (
    CalibrationState,
    Capability,
    QualityTier,
    SensorType,
    TargetClass,
)
from src.domain.observation import Observation
from src.domain.provenance import Provenance
from src.preprocessing.registration.bitemporal import (
    BitemporalRegistration,
    RegisteredPair,
)


def _dummy_provenance(prov_id: str = "prov-001") -> Provenance:
    return Provenance(
        provenance_id=prov_id,
        source_asset_id="asset-001",
        acquisition_metadata_ref="s3://sentinel/metadata.xml",
        pipeline_version="2.0.0",
        git_commit="abcdef123",
        config_hash="cfg-hash-001",
        model_id=None,
        model_version=None,
        input_checksum="input-chk-001",
        output_checksum="output-chk-001",
        processing_run_id="run-001",
        processing_time_utc=datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc),
    )


def _create_observation(
    obs_id: str = "obs-001",
    sensor: SensorType = SensorType.S2_MSI,
    product: str = "S2MSI1C",
    gsd_m: float = 10.0,
) -> Observation:
    poly = Polygon([(107.0, 15.0), (109.0, 15.0), (109.0, 17.0), (107.0, 17.0)])
    return Observation(
        observation_id=obs_id,
        source="CDSE",
        platform="Sentinel-2A" if sensor == SensorType.S2_MSI else "Sentinel-1A",
        sensor=sensor,
        product=product,
        processing_level="L1C",
        acquisition_time_utc=datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc),
        crs="EPSG:4326",
        ground_sample_distance_m=gsd_m,
        bounds_wgs84=poly,
        asset_uri=f"s3://sentinel/{obs_id}.safe",
        metadata_uri=f"s3://sentinel/{obs_id}_mtd.xml",
        quality_flags=("DATA_OK",),
        provenance=_dummy_provenance(f"prov-{obs_id}"),
    )


def test_incompatible_sensors_refused() -> None:
    reg = BitemporalRegistration()
    obs_s1 = _create_observation(
        obs_id="obs-s1", sensor=SensorType.S1_SAR, product="S1_GRD", gsd_m=10.0
    )
    obs_s2 = _create_observation(
        obs_id="obs-s2", sensor=SensorType.S2_MSI, product="S2MSI1C", gsd_m=10.0
    )
    arr_a = np.zeros((50, 50), dtype=np.float32)
    arr_b = np.zeros((50, 50), dtype=np.float32)

    with pytest.raises(
        ValueError, match="Incompatible observation sensors for bitemporal registration"
    ):
        reg.align(obs_s1, obs_s2, arr_a, arr_b)


def test_incompatible_products_refused() -> None:
    reg = BitemporalRegistration()
    obs_a = _create_observation(obs_id="obs-a", product="S2MSI1C")
    obs_b = _create_observation(obs_id="obs-b", product="S2MSI2A")
    arr = np.zeros((50, 50), dtype=np.float32)

    with pytest.raises(
        ValueError, match="Incompatible observation products for bitemporal registration"
    ):
        reg.align(obs_a, obs_b, arr, arr)


def test_incompatible_gsd_refused() -> None:
    reg = BitemporalRegistration()
    obs_a = _create_observation(obs_id="obs-a", gsd_m=10.0)
    obs_b = _create_observation(obs_id="obs-b", gsd_m=20.0)
    arr = np.zeros((50, 50), dtype=np.float32)

    with pytest.raises(
        ValueError, match="Incompatible observation resolutions for bitemporal registration"
    ):
        reg.align(obs_a, obs_b, arr, arr)


def test_high_residual_flags_georegistration_uncertain() -> None:
    reg = BitemporalRegistration(residual_threshold_px=0.5)
    obs_a = _create_observation(obs_id="obs-a")
    obs_b = _create_observation(obs_id="obs-b")

    arr_a = np.zeros((100, 100), dtype=np.float32)
    arr_a[30:50, 30:50] = 1.0

    arr_b = scipy.ndimage.shift(arr_a, (1.5, 1.5))

    pair = reg.align(obs_a, obs_b, arr_a, arr_b)

    assert pair.residual_px > 0.5
    assert pair.quality_tier == QualityTier.GEOREGISTRATION_UNCERTAIN


def test_change_detector_skips_misaligned_pair() -> None:
    reg = BitemporalRegistration(residual_threshold_px=0.5)
    obs_a = _create_observation(obs_id="obs-a")
    obs_b = _create_observation(obs_id="obs-b")

    arr_a = np.zeros((100, 100), dtype=np.float32)
    arr_a[30:50, 30:50] = 1.0
    arr_b = scipy.ndimage.shift(arr_a, (2.0, 2.0))

    pair = reg.align(obs_a, obs_b, arr_a, arr_b)
    assert pair.quality_tier == QualityTier.GEOREGISTRATION_UNCERTAIN

    detector = BitemporalChangeDetector()
    assert detector.capability == Capability.IMPLEMENTED

    detections = detector.detect(pair)
    assert len(detections) == 0
    assert isinstance(detections, SkippedLowQuality)
    assert detections.reason == "QualityTier.GEOREGISTRATION_UNCERTAIN"


def test_valid_change_detection_contract() -> None:
    reg = BitemporalRegistration(residual_threshold_px=0.5)
    obs_a = _create_observation(obs_id="obs-a")
    obs_b = _create_observation(obs_id="obs-b")

    arr_a = np.full((100, 100), 0.1, dtype=np.float32)
    arr_a[15:35, 15:35] = 0.5

    arr_b = arr_a.copy()
    arr_b[60:75, 60:75] = 0.95

    pair = reg.align(obs_a, obs_b, arr_a, arr_b)
    assert pair.residual_px <= 0.5
    assert pair.quality_tier == QualityTier.DATA_OK

    detector = BitemporalChangeDetector(
        diff_threshold=0.3,
        min_area_px=10,
        morph_kernel_size=3,
    )
    detections = detector.detect(pair)

    assert len(detections) >= 1
    det = detections[0]
    assert det.target_class == TargetClass.INFRASTRUCTURE_CHANGE
    assert det.sensor == SensorType.S2_MSI
    assert det.confidence.calibration_state == CalibrationState.NOT_CALIBRATED
    assert det.confidence.calibrated_probability is None
    assert det.confidence.quality_score == 1.0
    assert det.confidence.raw_score > 0.0
    assert det.provenance.pipeline_version == "2.0.0"
    assert det.provenance.source_asset_id == obs_b.observation_id
    assert det.geom_wgs84 is not None
    assert not det.geom_wgs84.is_empty


def test_morphological_opening_filters_single_pixel_noise() -> None:
    reg = BitemporalRegistration()
    obs_a = _create_observation(obs_id="obs-a")
    obs_b = _create_observation(obs_id="obs-b")

    arr_a = np.full((80, 80), 0.1, dtype=np.float32)
    arr_b = arr_a.copy()
    arr_b[10, 10] = 0.95
    arr_b[25, 40] = 0.95

    pair = reg.align(obs_a, obs_b, arr_a, arr_b)
    detector = BitemporalChangeDetector(
        diff_threshold=0.3,
        min_area_px=2,
        morph_kernel_size=3,
    )
    detections = detector.detect(pair)
    assert len(detections) == 0


def test_subpixel_alignment_corrects_small_shift() -> None:
    reg = BitemporalRegistration(residual_threshold_px=0.5)
    obs_a = _create_observation(obs_id="obs-a")
    obs_b = _create_observation(obs_id="obs-b")

    arr_a = np.zeros((80, 80), dtype=np.float32)
    arr_a[30:50, 30:50] = 1.0

    arr_b = scipy.ndimage.shift(arr_a, (0.2, 0.2))

    pair = reg.align(obs_a, obs_b, arr_a, arr_b)
    assert pair.residual_px <= 0.5
    assert pair.quality_tier == QualityTier.DATA_OK
    assert pair.aligned_arr_b.shape == arr_b.shape
