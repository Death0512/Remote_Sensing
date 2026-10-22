from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytest
from shapely.geometry import Polygon

from src.detection.experimental.parallax_detector import ParallaxInflightDetector
from src.detection.optical.aircraft_detector import (
    OpticalAircraftDetector,
    OpticalModelDetector,
    SkippedLowQuality,
)
from src.detection.optical.candidate_generator import (
    HeuristicBrightBlobCandidateGenerator,
)
from src.domain.confidence import ConfidenceVector
from src.domain.enums import (
    CalibrationState,
    Capability,
    EvidenceKind,
    QualityTier,
    RadiometricState,
    SensorType,
    TargetClass,
)
from src.domain.evidence import Evidence
from src.domain.observation import Observation
from src.domain.provenance import Provenance
from src.ingestion.providers.base import ProviderNotAvailableError
from src.preprocessing.optical.preprocessor import OpticalPreprocessor


def _dummy_provenance(prov_id: str = "prov-root-opt") -> Provenance:
    return Provenance(
        provenance_id=prov_id,
        source_asset_id="asset-opt-001",
        acquisition_metadata_ref="s3://sentinel2/l1c_meta.xml",
        pipeline_version="2.0.0",
        git_commit="abcdef0",
        config_hash="cfg-opt-01",
        model_id=None,
        model_version=None,
        input_checksum="input-chk-opt",
        output_checksum="output-chk-opt",
        processing_run_id="run-opt-init",
        processing_time_utc=datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc),
    )


def _create_observation(
    obs_id: str = "obs-opt-001",
    quality_flags: tuple[str, ...] = ("TOA_REFLECTANCE",),
) -> Observation:
    poly = Polygon([(107.0, 15.0), (109.0, 15.0), (109.0, 17.0), (107.0, 17.0)])
    return Observation(
        observation_id=obs_id,
        source="CDSE",
        platform="Sentinel-2A",
        sensor=SensorType.S2_MSI,
        product="S2MSI1C",
        processing_level="L1C",
        acquisition_time_utc=datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc),
        crs="EPSG:4326",
        ground_sample_distance_m=10.0,
        bounds_wgs84=poly,
        asset_uri="s3://sentinel2/tile.safe",
        metadata_uri="s3://sentinel2/mtd.xml",
        quality_flags=quality_flags,
        provenance=_dummy_provenance(),
    )


def test_optical_cloud_gate_skips_cloudy_scenes() -> None:
    preprocessor = OpticalPreprocessor(default_cloud_threshold=0.30)
    assert preprocessor.capability == Capability.IMPLEMENTED

    obs = _create_observation()
    cloud_mask = np.zeros((100, 100), dtype=bool)
    cloud_mask[:45, :] = True

    proc_obs = preprocessor.process(obs, cloud_mask=cloud_mask)
    assert proc_obs.quality_tier == QualityTier.CLOUD_CONTAMINATED
    assert proc_obs.radiometric_state == RadiometricState.TOA_REFLECTANCE

    detector = OpticalAircraftDetector(weights_path=None)
    synthetic_image = np.zeros((100, 100), dtype=np.float32)
    synthetic_image[50:65, 50:55] = 0.95

    result = detector.detect(proc_obs, synthetic_image)
    assert isinstance(result, SkippedLowQuality)
    assert len(result) == 0
    assert result.reason == "QualityTier.CLOUD_CONTAMINATED"


def test_heuristic_bright_blob_truth_in_advertising() -> None:
    generator = HeuristicBrightBlobCandidateGenerator(
        threshold=0.5,
        min_area_px=4,
        max_area_px=500,
        pixel_spacing_m=1.0,
    )
    assert generator.capability == Capability.HEURISTIC
    assert generator.__class__.__name__ == "HeuristicBrightBlobCandidateGenerator"
    assert "yolo" not in generator.__class__.__name__.lower()

    obs = _create_observation()
    preprocessor = OpticalPreprocessor()
    proc_obs = preprocessor.process(obs, cloud_mask=None)

    image = np.zeros((80, 80), dtype=np.float32)
    image[20:35, 20:25] = 0.9

    candidates = generator.generate(image, processed_observation=proc_obs)
    assert len(candidates) >= 1
    for cand in candidates:
        assert cand.generator == "HeuristicBrightBlobCandidateGenerator"
        assert cand.generator_capability == Capability.HEURISTIC
        feat_dict = dict(cand.features)
        assert "brightness_max" in feat_dict
        assert "length_m" in feat_dict
        assert "width_m" in feat_dict
        assert "aspect_ratio" in feat_dict


def test_optical_detector_uncalibrated_confidence() -> None:
    detector = OpticalAircraftDetector(
        weights_path=None,
        min_length_m=5.0,
        max_length_m=80.0,
        pixel_spacing_m=1.0,
    )
    assert detector.capability == Capability.HEURISTIC

    obs = _create_observation()
    preprocessor = OpticalPreprocessor()
    proc_obs = preprocessor.process(obs, cloud_mask=None)
    assert proc_obs.quality_tier == QualityTier.DATA_OK

    image = np.zeros((100, 100), dtype=np.float32)
    image[30:50, 30:35] = 0.9

    detections = detector.detect(proc_obs, image)
    assert isinstance(detections, tuple)
    assert len(detections) >= 1
    for det in detections:
        assert det.target_class == TargetClass.AIRCRAFT
        assert isinstance(det.confidence, ConfidenceVector)
        assert det.confidence.calibrated_probability is None
        assert det.confidence.calibration_state == CalibrationState.NOT_CALIBRATED
        assert det.confidence.model_score is None
        assert det.confidence.raw_score > 0.0


def test_parallax_detector_emits_experimental_evidence_not_alerts() -> None:
    detector = ParallaxInflightDetector(
        pixel_spacing_m=10.0,
        inter_band_delay_s=0.5,
        min_displacement_px=2.0,
        max_displacement_px=25.0,
        triplet_colinearity_max_deg=20.0,
        min_velocity_kmh=250.0,
        max_velocity_kmh=1200.0,
        peak_threshold=0.7,
    )
    assert detector.capability == Capability.HEURISTIC

    obs = _create_observation()
    preprocessor = OpticalPreprocessor()
    proc_obs = preprocessor.process(obs)

    rgb = np.zeros((3, 100, 100), dtype=np.float32)
    rgb[0, 50, 45] = 0.95
    rgb[1, 50, 50] = 0.95
    rgb[2, 50, 55] = 0.95

    evidences = detector.detect(
        processed_observation=proc_obs,
        image_rgb=rgb,
        pixel_spacing_m=10.0,
        inter_band_delay_s=0.5,
    )

    assert isinstance(evidences, tuple)
    assert len(evidences) >= 1
    for ev in evidences:
        assert isinstance(ev, Evidence)
        assert ev.kind == EvidenceKind.KINEMATICS
        assert 500.0 <= ev.value <= 900.0
        assert ev.supports == "EXPERIMENTAL_INFLIGHT_TARGET"
        assert not hasattr(ev, "priority")
        assert not hasattr(ev, "alert_id")


def test_optical_model_detector_missing_weights_raises() -> None:
    with pytest.raises(ProviderNotAvailableError):
        OpticalModelDetector(weights_path=Path("/tmp/nonexistent_yolov8_model.pt"))
