from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from shapely.geometry import Polygon

from src.cli.main import cli_entrypoint
from src.detection.optical.aircraft_detector import (
    HeuristicOpticalAircraftDetector,
    OpticalAircraftDetector,
    OpticalModelDetector,
    OpticalModelUnavailableError,
    SkippedLowQuality,
    YoloObbAircraftDetector,
)
from src.detection.optical.candidate_generator import (
    HeuristicBrightBlobCandidateGenerator,
)
from src.domain.confidence import ConfidenceVector
from src.domain.enums import (
    CalibrationState,
    Capability,
    QualityTier,
    RadiometricState,
    SensorType,
    TargetClass,
)
from src.domain.geometry import (
    AxisAlignedBoundingBox,
    OrientedBoundingBox,
)
from src.domain.observation import Observation, ProcessedObservation
from src.domain.provenance import Provenance
from src.ingestion.providers.base import ProviderNotAvailableError
from src.preprocessing.optical.preprocessor import OpticalPreprocessor
from src.provenance.exceptions import ModelUnavailableError


def _make_provenance(prov_id: str = "prov-opt-v3") -> Provenance:
    return Provenance(
        provenance_id=prov_id,
        source_asset_id="asset-opt-v3-001",
        acquisition_metadata_ref="s3://sentinel2/mtd.xml",
        pipeline_version="3.0.0",
        git_commit="abcdef0",
        config_hash="cfghash-v3",
        model_id=None,
        model_version=None,
        input_checksum="inchk-v3",
        output_checksum="outchk-v3",
        processing_run_id="run-opt-v3",
        processing_time_utc=datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc),
    )


def _make_observation(
    obs_id: str = "obs-opt-v3",
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
        provenance=_make_provenance(),
    )


def test_aabb_geometry_contract_integrity() -> None:
    box = AxisAlignedBoundingBox(min_col=10.0, min_row=20.0, max_col=50.0, max_row=80.0)
    assert box.min_col == 10.0
    assert box.min_row == 20.0
    assert box.max_col == 50.0
    assert box.max_row == 80.0
    assert box.width == 40.0
    assert box.height == 60.0
    assert box.cx == 30.0
    assert box.cy == 50.0
    assert box.area == 2400.0
    assert box.centroid == (30.0, 50.0)
    assert box.to_tuple() == (10.0, 20.0, 50.0, 80.0)
    assert len(box) == 4
    assert box[0] == 10.0
    assert tuple(box) == (10.0, 20.0, 50.0, 80.0)

    inverted = AxisAlignedBoundingBox(min_col=50.0, min_row=80.0, max_col=10.0, max_row=20.0)
    assert inverted.min_col == 10.0
    assert inverted.max_col == 50.0
    assert inverted.min_row == 20.0
    assert inverted.max_row == 80.0

    obb = box.to_obb()
    assert isinstance(obb, OrientedBoundingBox)
    assert obb.cx == box.cx
    assert obb.cy == box.cy
    assert obb.width == box.width
    assert obb.height == box.height
    assert obb.angle_deg == 0.0

    poly = box.to_polygon()
    assert poly.area == box.area

    from_tup = AxisAlignedBoundingBox.from_tuple((5.0, 10.0, 25.0, 30.0))
    assert from_tup.width == 20.0
    assert from_tup.height == 20.0

    with pytest.raises(FrozenInstanceError):
        box.min_col = 0.0


def test_obb_geometry_contract_integrity() -> None:
    obb = OrientedBoundingBox(cx=100.0, cy=100.0, width=40.0, height=20.0, angle_deg=90.0)
    assert obb.cx == 100.0
    assert obb.cy == 100.0
    assert obb.width == 40.0
    assert obb.height == 20.0
    assert obb.angle_deg == 90.0
    assert obb.area == 800.0
    assert obb.centroid == (100.0, 100.0)
    assert len(obb) == 5
    assert obb[0] == 100.0
    assert tuple(obb) == (100.0, 100.0, 40.0, 20.0, 90.0)

    norm_angle = OrientedBoundingBox(cx=0.0, cy=0.0, width=10.0, height=10.0, angle_deg=450.0)
    assert norm_angle.angle_deg == 90.0

    aabb = obb.to_aabb()
    assert isinstance(aabb, AxisAlignedBoundingBox)
    assert aabb.width == pytest.approx(20.0, abs=1e-3)
    assert aabb.height == pytest.approx(40.0, abs=1e-3)
    assert aabb.cx == pytest.approx(100.0, abs=1e-3)
    assert aabb.cy == pytest.approx(100.0, abs=1e-3)

    poly = obb.to_polygon()
    assert poly.area == pytest.approx(obb.area, abs=1e-3)

    from_xywhr = OrientedBoundingBox.from_xywhr(10.0, 20.0, 30.0, 15.0, math.pi / 2.0)
    assert from_xywhr.angle_deg == pytest.approx(90.0, abs=1e-3)

    from_tup = OrientedBoundingBox.from_tuple((1.0, 2.0, 3.0, 4.0, 5.0))
    assert from_tup.to_tuple() == (1.0, 2.0, 3.0, 4.0, 5.0)

    with pytest.raises(FrozenInstanceError):
        obb.cx = 0.0


def test_optical_model_detector_missing_weights_fails_closed() -> None:
    nonexistent = Path("/nonexistent/data/weights/yolov8n-obb.pt")

    with pytest.raises(OpticalModelUnavailableError) as exc_info:
        OpticalModelDetector(weights_path=nonexistent, strict=True)

    err = exc_info.value
    assert isinstance(err, ModelUnavailableError)
    assert isinstance(err, ProviderNotAvailableError)

    detector_lenient = OpticalModelDetector(weights_path=nonexistent, strict=False)
    assert detector_lenient.capability == Capability.UNAVAILABLE

    proc_obs = ProcessedObservation(
        processed_id="proc-test",
        observation_id="obs-test",
        preprocessing_profile="standard",
        radiometric_state=RadiometricState.TOA_REFLECTANCE,
        geometric_state="GEOCODED",
        masks=(),
        quality_tier=QualityTier.DATA_OK,
        provenance=_make_provenance(),
    )
    with pytest.raises(OpticalModelUnavailableError):
        detector_lenient.detect(proc_obs, np.zeros((32, 32), dtype=np.float32))


def test_heuristic_detector_honesty_and_rejection_of_weights() -> None:
    with pytest.raises(OpticalModelUnavailableError):
        OpticalAircraftDetector(weights_path="/some/weights.pt")

    detector = OpticalAircraftDetector(
        weights_path=None,
        min_length_m=5.0,
        max_length_m=80.0,
        pixel_spacing_m=1.0,
    )
    assert detector.capability == Capability.HEURISTIC
    assert HeuristicOpticalAircraftDetector is OpticalAircraftDetector

    obs = _make_observation()
    preprocessor = OpticalPreprocessor()
    proc_obs = preprocessor.process(obs, cloud_mask=None)

    image = np.zeros((100, 100), dtype=np.float32)
    image[30:50, 30:35] = 0.95

    detections = detector.detect(proc_obs, image)
    assert isinstance(detections, tuple)
    assert len(detections) >= 1

    for det in detections:
        assert det.target_class == TargetClass.AIRCRAFT
        assert isinstance(det.bounding_box, AxisAlignedBoundingBox)
        assert len(det.geom_px) == 4
        assert det.confidence.calibration_state == CalibrationState.NOT_CALIBRATED
        assert det.confidence.calibrated_probability is None
        assert det.confidence.model_score is None
        assert det.confidence.raw_score > 0.0
        assert det.provenance.pipeline_version == "3.0.0"
        attr_dict = dict(det.attributes)
        assert attr_dict.get("geometry_type") == "AABB"


def test_cloud_gating_blocks_both_heuristic_and_model_detectors() -> None:
    obs = _make_observation()
    preprocessor = OpticalPreprocessor()
    cloud_mask = np.ones((50, 50), dtype=bool)
    proc_obs = preprocessor.process(obs, cloud_mask=cloud_mask)
    assert proc_obs.quality_tier == QualityTier.CLOUD_CONTAMINATED

    heuristic_detector = OpticalAircraftDetector()
    h_result = heuristic_detector.detect(proc_obs, np.zeros((50, 50), dtype=np.float32))
    assert isinstance(h_result, SkippedLowQuality)
    assert len(h_result) == 0
    assert h_result.reason == "QualityTier.CLOUD_CONTAMINATED"

    with patch("src.detection.optical.aircraft_detector.ModelArtifactValidator") as mock_val_cls, patch("ultralytics.YOLO"):
        mock_val = MagicMock()
        mock_val.validate_and_resolve.return_value = Path("/tmp/mock_yolo.pt")
        mock_val_cls.return_value = mock_val

        model_detector = OpticalModelDetector(weights_path="/tmp/mock_yolo.pt")
        m_result = model_detector.detect(proc_obs, np.zeros((50, 50), dtype=np.float32))
        assert isinstance(m_result, SkippedLowQuality)
        assert len(m_result) == 0
        assert m_result.reason == "QualityTier.CLOUD_CONTAMINATED"


def test_model_detector_obb_output_honesty() -> None:
    assert YoloObbAircraftDetector is OpticalModelDetector

    with patch("src.detection.optical.aircraft_detector.ModelArtifactValidator") as mock_val_cls, patch("ultralytics.YOLO") as mock_yolo_cls:
        mock_val = MagicMock()
        mock_val.validate_and_resolve.return_value = Path("/tmp/mock_yolo.pt")
        mock_val_cls.return_value = mock_val

        mock_box = MagicMock()
        mock_box.xywhr = np.array([50.0, 50.0, 30.0, 15.0, 0.785398], dtype=np.float32)
        mock_box.conf = np.array([0.88], dtype=np.float32)

        mock_res = MagicMock()
        mock_res.obb = None
        mock_res.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.return_value = [mock_res]
        mock_yolo_cls.return_value = mock_model

        detector = OpticalModelDetector(weights_path="/tmp/mock_yolo.pt")
        assert detector.capability == Capability.EXTERNAL_MODEL

        obs = _make_observation()
        preprocessor = OpticalPreprocessor()
        proc_obs = preprocessor.process(obs, cloud_mask=None)

        results = detector.detect(proc_obs, np.zeros((100, 100), dtype=np.float32))
        assert isinstance(results, tuple)
        assert len(results) == 1
        det = results[0]
        assert det.target_class == TargetClass.AIRCRAFT
        assert isinstance(det.bounding_box, OrientedBoundingBox)
        assert det.bounding_box.cx == 50.0
        assert det.bounding_box.cy == 50.0
        assert det.bounding_box.width == 30.0
        assert det.bounding_box.height == 15.0
        assert det.bounding_box.angle_deg == pytest.approx(45.0, abs=1e-1)
        assert len(det.geom_px) == 5
        attr_dict = dict(det.attributes)
        assert attr_dict.get("geometry_type") == "OBB"
        assert det.confidence.model_score == pytest.approx(0.88, abs=1e-2)
        assert det.provenance.pipeline_version == "3.0.0"


def test_cli_execution_on_optical_sensors(tmp_path: Path) -> None:
    img = np.zeros((100, 100), dtype=np.float32)
    img[40:45, 40:43] = 0.95
    img_path = tmp_path / "optical_scene.npy"
    np.save(str(img_path), img)

    out_dir = tmp_path / "out_cli_optical"
    ret = cli_entrypoint([
        "scan",
        "--image", str(img_path),
        "--sensor", "S2_MSI",
        "--detector", "heuristic",
        "--output-dir", str(out_dir),
    ])
    assert ret == 0

    alerts_file = out_dir / "alerts.geojson"
    assert alerts_file.exists()
    with open(alerts_file, encoding="utf-8") as f:
        data = json.load(f)

    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) >= 1
    feat = data["features"][0]
    assert feat["properties"]["sensor"] == "S2_MSI"
    assert feat["properties"]["confidence"]["raw_score"] > 0.0
    assert feat["properties"]["confidence"]["calibrated_probability"] is None

    out_dir_model = tmp_path / "out_cli_model"
    with patch("src.cli.main.OpticalModelDetector") as mock_det_cls:
        mock_det = MagicMock()
        mock_det.detect.return_value = ()
        mock_det_cls.return_value = mock_det
        ret_model = cli_entrypoint([
            "scan",
            "--image", str(img_path),
            "--sensor", "OPTICAL",
            "--detector", "model",
            "--output-dir", str(out_dir_model),
        ])
        assert ret_model == 0
        assert mock_det.detect.called
