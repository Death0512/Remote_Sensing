from datetime import datetime, timezone
import math
import numpy as np
import pytest
from shapely.geometry import Polygon

from src.domain.enums import (
    GeometricState,
    GeometryStatus,
    QualityTier,
    RadiometricState,
    SensorType,
)
from src.domain.geometry import GeodeticGeometry, MapGeometry, PixelGeometry
from src.domain.observation import Observation, ProcessedObservation
from src.domain.provenance import Provenance
from src.detection.experimental.parallax_detector import (
    ParallaxInflightDetector,
    ParallaxQuarantineError,
)
from src.preprocessing.calibration.sar import (
    SarCalibrator,
    calibrate_sar_digital_numbers,
)
from src.preprocessing.calibration.optical import (
    OpticalCalibrator,
    calibrate_optical_digital_numbers,
)
from src.preprocessing.georeferencing.pipeline import (
    GeoreferencingPipeline,
    GeoreferenceValidationError,
)
from src.preprocessing.masking.scl import (
    SclClass,
    SclMaskExtractor,
    evaluate_scl_cloud_coverage,
    extract_scl_masks,
)
from src.preprocessing.optical.preprocessor import OpticalPreprocessor
from src.preprocessing.sar.preprocessor import SarPreprocessor


def _sample_provenance() -> Provenance:
    return Provenance(
        provenance_id="prov-test-p4-001",
        source_asset_id="asset-test-p4-001",
        acquisition_metadata_ref="meta-test-p4-001",
        pipeline_version="3.0.0",
        git_commit="6f84f43",
        config_hash="sha256:cfgtest",
        model_id=None,
        model_version=None,
        input_checksum="insha256",
        output_checksum="outsha256",
        processing_run_id="run-test-p4-001",
        processing_time_utc=datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc),
    )


def _sample_observation(
    sensor: SensorType = SensorType.S1_SAR,
    processing_level: str = "GRD",
    quality_flags: tuple[str, ...] = ("OK",),
    geometry_status: GeometryStatus = GeometryStatus.GEOCODED,
) -> Observation:
    poly = Polygon([(108.0, 16.0), (108.1, 16.0), (108.1, 16.1), (108.0, 16.1)])
    return Observation(
        observation_id="obs-test-001",
        source="CDSE",
        platform="Sentinel-1A",
        sensor=sensor,
        product="GRD",
        processing_level=processing_level,
        acquisition_time_utc=datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc),
        crs="EPSG:4326",
        ground_sample_distance_m=10.0,
        bounds_wgs84=poly,
        asset_uri="s3://asset",
        metadata_uri="s3://meta",
        quality_flags=quality_flags,
        provenance=_sample_provenance(),
        geometry_status=geometry_status,
    )


def test_sar_calibration_dn_to_sigma0() -> None:
    calibrator = SarCalibrator()
    dn = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)
    lut = np.array([[10.0, 10.0], [10.0, 10.0]], dtype=np.float32)

    sigma0_db, state = calibrator.calibrate_dn_to_sigma0(dn, lut, input_domain="dn", to_db=True)
    assert state == RadiometricState.CALIBRATED_SIGMA0
    assert sigma0_db.shape == (2, 2)
    assert math.isclose(sigma0_db[0, 0], 10.0 * math.log10(100.0), abs_tol=1e-4)

    sigma0_lin, state_lin = calibrator.calibrate_dn_to_sigma0(dn, lut, input_domain="dn", to_db=False)
    assert state_lin == RadiometricState.CALIBRATED_SIGMA0
    assert math.isclose(sigma0_lin[0, 0], 100.0, abs_tol=1e-4)


def test_sar_calibration_dn_to_gamma0() -> None:
    calibrator = SarCalibrator()
    dn = np.array([200.0], dtype=np.float32)
    lut = np.array([10.0], dtype=np.float32)
    inc_rad = np.array([math.radians(60.0)], dtype=np.float32)

    gamma0_db, state = calibrator.calibrate_dn_to_gamma0(
        dn, lut, incidence_angle_rad=inc_rad, input_domain="dn", to_db=True
    )
    assert state == RadiometricState.CALIBRATED_GAMMA0
    expected_lin = (200.0 ** 2 / 10.0 ** 2) / math.cos(math.radians(60.0))
    expected_db = 10.0 * math.log10(expected_lin)
    assert math.isclose(gamma0_db[0], expected_db, abs_tol=1e-4)


def test_sar_prohibits_amplitude_mixing() -> None:
    calibrator = SarCalibrator()
    dn = np.array([[10.0]], dtype=np.float32)
    lut = np.array([[1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="Prohibited mixing: amplitude"):
        calibrator.calibrate_dn_to_sigma0(dn, lut, input_domain="amplitude")

    with pytest.raises(ValueError, match="Prohibited mixing"):
        calibrator.calibrate_dn_to_gamma0(dn, lut, input_domain="invalid_type")


def test_optical_calibration_toa_and_surface() -> None:
    calibrator = OpticalCalibrator(default_scale=10000.0)
    dn = np.array([[2000.0, 5000.0]], dtype=np.float32)

    refl_toa, state_toa = calibrator.dn_to_reflectance(dn, processing_level="L1C")
    assert state_toa == RadiometricState.TOA_REFLECTANCE
    assert math.isclose(refl_toa[0, 0], 0.2, abs_tol=1e-5)
    assert math.isclose(refl_toa[0, 1], 0.5, abs_tol=1e-5)

    refl_boa, state_boa = calibrator.dn_to_reflectance(dn, processing_level="L2A")
    assert state_boa == RadiometricState.SURFACE_REFLECTANCE
    assert math.isclose(refl_boa[0, 0], 0.2, abs_tol=1e-5)


def test_scl_mask_extraction_and_quality_gating() -> None:
    scl = np.zeros((10, 10), dtype=np.uint8)
    scl[0:2, :] = int(SclClass.CLOUD_HIGH_PROBABILITY)
    scl[2:4, :] = int(SclClass.CLOUD_SHADOWS)
    scl[4:6, :] = int(SclClass.THIN_CIRRUS)
    scl[6:, :] = int(SclClass.WATER)

    masks = extract_scl_masks(scl)
    assert masks["cloud_mask"].shape == (10, 10)
    assert np.all(masks["cloud_mask"][0:2, :])
    assert np.all(masks["shadow_mask"][2:4, :])
    assert np.all(masks["cirrus_mask"][4:6, :])

    tier_clean, frac_clean, _ = evaluate_scl_cloud_coverage(scl, cloud_threshold=0.70)
    assert tier_clean == QualityTier.DATA_OK

    tier_cloudy, frac_cloudy, stats = evaluate_scl_cloud_coverage(scl, cloud_threshold=0.30)
    assert tier_cloudy == QualityTier.CLOUD_CONTAMINATED
    assert frac_cloudy == 0.6


def test_optical_preprocessor_with_scl() -> None:
    obs = _sample_observation(sensor=SensorType.S2_MSI, processing_level="L2A")
    preproc = OpticalPreprocessor(default_cloud_threshold=0.20)

    clean_scl = np.full((10, 10), int(SclClass.VEGETATION), dtype=np.uint8)
    proc_clean = preproc.process(obs, scl_data=clean_scl)
    assert proc_clean.quality_tier == QualityTier.DATA_OK
    assert proc_clean.radiometric_state == RadiometricState.SURFACE_REFLECTANCE

    cloudy_scl = np.full((10, 10), int(SclClass.CLOUD_HIGH_PROBABILITY), dtype=np.uint8)
    proc_cloudy = preproc.process(obs, scl_data=cloudy_scl)
    assert proc_cloudy.quality_tier == QualityTier.CLOUD_CONTAMINATED


def test_georeferencing_pipeline_roundtrip() -> None:
    pipeline = GeoreferencingPipeline()
    affine = (10.0, 0.0, 500000.0, 0.0, -10.0, 1800000.0)
    crs = "EPSG:32648"

    pipeline.configure(crs=crs, affine=affine)
    assert pipeline.is_valid

    px = PixelGeometry(row=128.0, col=256.0, shape=(512, 512))
    geo, status = pipeline.forward(px, validate_roundtrip=True)
    assert status == GeometryStatus.GEOCODED
    assert isinstance(geo, GeodeticGeometry)

    res_px = pipeline.evaluate_roundtrip_error(px, geo)
    assert res_px < 0.5

    back_px = pipeline.backward(geo, shape=(512, 512))
    assert math.isclose(back_px.col, 256.0, abs_tol=1e-3)
    assert math.isclose(back_px.row, 128.0, abs_tol=1e-3)


def test_georeferencing_pipeline_unknown_and_forbid_untransformed() -> None:
    unconfigured = GeoreferencingPipeline()
    px = PixelGeometry(row=100.0, col=200.0, shape=(512, 512))

    geo, status = unconfigured.forward(px, validate_roundtrip=False)
    assert status == GeometryStatus.UNKNOWN

    with pytest.raises(GeoreferenceValidationError):
        unconfigured.pixel_to_geo(200.0, 100.0)

    with pytest.raises(GeoreferenceValidationError):
        unconfigured.geo_to_pixel(108.0, 16.0)

    identity_pipeline = GeoreferencingPipeline()
    with pytest.raises(GeoreferenceValidationError, match="unreferenced identity"):
        identity_pipeline.configure(crs="EPSG:4326", affine=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0))


def test_parallax_quarantine_enforced() -> None:
    detector = ParallaxInflightDetector()
    obs = _sample_observation(sensor=SensorType.S2_MSI)
    preproc = OpticalPreprocessor()
    proc = preproc.process(obs)
    rgb = np.ones((100, 100, 3), dtype=np.float32)

    with pytest.raises(ParallaxQuarantineError, match="QUARANTINED"):
        detector.detect(proc, rgb, sensor_metadata=None)

    with pytest.raises(ParallaxQuarantineError, match="QUARANTINED"):
        detector.detect(proc, rgb, sensor_metadata={"camera": "rgb_render"})

    valid_meta = {
        "line_time": 0.001,
        "acquisition_time_utc": "2026-10-22T12:00:00Z",
        "inter_band_delay_s": 0.52,
    }
    results = detector.detect(proc, rgb, sensor_metadata=valid_meta)
    assert isinstance(results, tuple)
