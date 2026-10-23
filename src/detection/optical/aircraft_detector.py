from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
from uuid import uuid4
import numpy as np
from shapely.geometry import Point, Polygon

from src.detection.optical.candidate_generator import (
    HeuristicBrightBlobCandidateGenerator,
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
from src.domain.geometry import AxisAlignedBoundingBox, OrientedBoundingBox
from src.domain.observation import ProcessedObservation
from src.domain.provenance import Provenance
from src.ingestion.providers.base import ProviderNotAvailableError
from src.provenance.engine import ModelArtifactValidator
from src.provenance.exceptions import (
    ModelChecksumMismatchError,
    ModelUnavailableError,
)


class OpticalModelUnavailableError(ModelUnavailableError, ProviderNotAvailableError):
    pass


class SkippedLowQuality(tuple):
    def __new__(cls, reason: str = "QualityTier.CLOUD_CONTAMINATED"):
        instance = super().__new__(cls, ())
        object.__setattr__(instance, "reason", reason)
        return instance

    def __init__(self, reason: str = "QualityTier.CLOUD_CONTAMINATED") -> None:
        self.reason = reason


DetectionResult = Union[Tuple[Detection, ...], SkippedLowQuality]


def _extract_obb_boxes(res: Any) -> List[Tuple[OrientedBoundingBox, float]]:
    extracted: List[Tuple[OrientedBoundingBox, float]] = []
    obb_container = getattr(res, "obb", None)
    if obb_container is not None:
        xywhr_data = getattr(obb_container, "xywhr", None)
        conf_data = getattr(obb_container, "conf", None)
        if xywhr_data is not None:
            if hasattr(xywhr_data, "cpu"):
                xywhr_data = xywhr_data.cpu().numpy()
            if hasattr(conf_data, "cpu"):
                conf_data = conf_data.cpu().numpy()
            xywhr_arr = np.asarray(xywhr_data, dtype=np.float32)
            conf_arr = (
                np.asarray(conf_data, dtype=np.float32)
                if conf_data is not None
                else np.ones(len(xywhr_arr), dtype=np.float32)
            )
            if xywhr_arr.ndim == 1 and len(xywhr_arr) >= 5:
                xywhr_arr = xywhr_arr.reshape(1, -1)
                conf_arr = conf_arr.reshape(-1)
            for i in range(len(xywhr_arr)):
                row = xywhr_arr[i]
                score = float(conf_arr[i]) if i < len(conf_arr) else 1.0
                cx = float(row[0])
                cy = float(row[1])
                w = float(row[2])
                h = float(row[3])
                r = float(row[4])
                angle_deg = float(math.degrees(r))
                extracted.append(
                    (
                        OrientedBoundingBox(
                            cx=cx,
                            cy=cy,
                            width=w,
                            height=h,
                            angle_deg=angle_deg,
                        ),
                        score,
                    )
                )
            return extracted

    boxes = getattr(res, "boxes", None)
    if boxes is not None:
        for box in boxes:
            if hasattr(box, "xywhr"):
                xywhr = box.xywhr
                if hasattr(xywhr, "cpu"):
                    xywhr = xywhr.cpu().numpy()
                xywhr_flat = np.asarray(xywhr, dtype=np.float32).flatten()
                conf_val = getattr(box, "conf", 1.0)
                if hasattr(conf_val, "cpu"):
                    conf_val = conf_val.cpu().numpy()
                score_arr = np.asarray(conf_val).flatten()
                score = float(score_arr[0]) if score_arr.size > 0 else 1.0
                cx = float(xywhr_flat[0])
                cy = float(xywhr_flat[1])
                w = float(xywhr_flat[2])
                h = float(xywhr_flat[3])
                r = float(xywhr_flat[4])
                angle_deg = float(math.degrees(r))
                extracted.append(
                    (
                        OrientedBoundingBox(
                            cx=cx,
                            cy=cy,
                            width=w,
                            height=h,
                            angle_deg=angle_deg,
                        ),
                        score,
                    )
                )
    return extracted


class OpticalModelDetector:
    capability: Capability = Capability.EXTERNAL_MODEL

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = 0.25,
        model_name: str = "yolov8_obb",
        registry_path: Optional[Union[str, Path]] = None,
        strict: bool = True,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.model_name = model_name
        self.model = None
        self.weights_path: Optional[Path] = None

        try:
            validator = ModelArtifactValidator(registry_path=registry_path)
            resolved = validator.validate_and_resolve(
                model_name=model_name,
                override_path=weights_path,
            )
            self.weights_path = Path(resolved)
        except (
            ModelUnavailableError,
            ModelChecksumMismatchError,
            FileNotFoundError,
            Exception,
        ) as exc:
            if strict:
                raise OpticalModelUnavailableError(str(exc)) from exc
            self.capability = Capability.UNAVAILABLE
            return

        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.weights_path))
            self.capability = Capability.EXTERNAL_MODEL
        except Exception as exc:
            if strict:
                raise OpticalModelUnavailableError(
                    f"Failed to load YOLO model: {exc}"
                ) from exc
            self.capability = Capability.UNAVAILABLE

    def detect(
        self,
        processed_observation: ProcessedObservation,
        image: np.ndarray,
        pixel_spacing_m: float = 0.5,
        affine_transform: Optional[Any] = None,
        acquisition_time_utc: Optional[datetime] = None,
    ) -> DetectionResult:
        if self.capability == Capability.UNAVAILABLE or self.model is None:
            raise OpticalModelUnavailableError(
                "Model weights are unavailable; refusing to execute without verified weights"
            )

        if (
            processed_observation.quality_tier
            == QualityTier.CLOUD_CONTAMINATED
        ):
            return SkippedLowQuality("QualityTier.CLOUD_CONTAMINATED")

        results = self.model(image, conf=self.conf_threshold)
        detections: List[Detection] = []

        for res in results:
            extracted_boxes = _extract_obb_boxes(res)
            for obb, score in extracted_boxes:
                if affine_transform is not None and hasattr(
                    affine_transform, "pixel_to_geo"
                ):
                    geo_pts = [
                        affine_transform.pixel_to_geo(c[0], c[1])
                        for c in obb.corners
                    ]
                    geom_wgs84 = Polygon(geo_pts)
                else:
                    geom_wgs84 = obb.to_polygon()

                conf = ConfidenceVector(
                    raw_score=score,
                    model_score=score,
                    calibrated_probability=None,
                    uncertainty=None,
                    quality_score=1.0,
                    calibration_state=CalibrationState.NOT_CALIBRATED,
                )
                det_prov = Provenance(
                    provenance_id=f"prov-det-opt-{uuid4().hex[:12]}",
                    source_asset_id=processed_observation.provenance.source_asset_id,
                    acquisition_metadata_ref=processed_observation.provenance.acquisition_metadata_ref,
                    pipeline_version="3.0.0",
                    git_commit=processed_observation.provenance.git_commit,
                    config_hash=processed_observation.provenance.config_hash,
                    model_id=(
                        str(self.weights_path.name)
                        if self.weights_path
                        else self.model_name
                    ),
                    model_version="1.0",
                    input_checksum=processed_observation.provenance.output_checksum,
                    output_checksum=hashlib.sha256(
                        f"{obb.cx}_{obb.cy}_{score}_{obb.angle_deg}".encode(
                            "utf-8"
                        )
                    ).hexdigest()[:16],
                    processing_run_id=processed_observation.provenance.processing_run_id,
                    processing_time_utc=datetime.now(timezone.utc),
                    parent_provenance_id=processed_observation.provenance.provenance_id,
                )
                det = Detection(
                    detection_id=f"det-opt-{uuid4().hex[:12]}",
                    candidate_id=f"cand-model-{uuid4().hex[:12]}",
                    sensor=SensorType.VHR_OPTICAL,
                    geom_wgs84=geom_wgs84,
                    geom_px=obb.to_tuple(),
                    crs="EPSG:4326",
                    target_class=TargetClass.AIRCRAFT,
                    confidence=conf,
                    attributes=(
                        ("score", float(score)),
                        ("geometry_type", "OBB"),
                        ("cx", float(obb.cx)),
                        ("cy", float(obb.cy)),
                        ("width", float(obb.width)),
                        ("height", float(obb.height)),
                        ("angle_deg", float(obb.angle_deg)),
                    ),
                    acquisition_time_utc=acquisition_time_utc
                    or datetime.now(timezone.utc),
                    provenance=det_prov,
                    bounding_box=obb,
                )
                detections.append(det)

        return tuple(detections)


class OpticalAircraftDetector:
    capability: Capability = Capability.HEURISTIC

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        candidate_generator: Optional[
            HeuristicBrightBlobCandidateGenerator
        ] = None,
        min_length_m: float = 8.0,
        max_length_m: float = 85.0,
        min_aspect_ratio: float = 0.8,
        max_aspect_ratio: float = 6.0,
        pixel_spacing_m: float = 0.5,
    ) -> None:
        if weights_path is not None:
            raise OpticalModelUnavailableError(
                "OpticalAircraftDetector is a heuristic detector and does not accept weights. "
                "Use OpticalModelDetector for deep learning models."
            )
        self.min_length_m = min_length_m
        self.max_length_m = max_length_m
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.pixel_spacing_m = pixel_spacing_m
        self.capability = Capability.HEURISTIC
        self.candidate_generator = (
            candidate_generator
            or HeuristicBrightBlobCandidateGenerator(
                pixel_spacing_m=pixel_spacing_m
            )
        )

    def detect(
        self,
        processed_observation: ProcessedObservation,
        image: np.ndarray,
        pixel_spacing_m: Optional[float] = None,
        affine_transform: Optional[Any] = None,
        acquisition_time_utc: Optional[datetime] = None,
    ) -> DetectionResult:
        if (
            processed_observation.quality_tier
            == QualityTier.CLOUD_CONTAMINATED
        ):
            return SkippedLowQuality("QualityTier.CLOUD_CONTAMINATED")

        spacing = (
            pixel_spacing_m
            if pixel_spacing_m is not None
            else self.pixel_spacing_m
        )

        candidates = self.candidate_generator.generate(
            image=image,
            processed_observation=processed_observation,
            pixel_spacing_m=spacing,
            affine_transform=affine_transform,
        )

        detections: List[Detection] = []
        for cand in candidates:
            feat_dict = dict(cand.features)
            length_m = feat_dict.get("length_m", 0.0)
            width_m = feat_dict.get("width_m", 0.0)
            aspect_ratio = feat_dict.get("aspect_ratio", 1.0)
            orientation_deg = feat_dict.get("orientation_deg", 0.0)
            raw_score = feat_dict.get("brightness_max", 0.5)

            if not (self.min_length_m <= length_m <= self.max_length_m):
                continue
            if not (
                self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio
            ):
                continue

            quality_score = (
                1.0
                if processed_observation.quality_tier == QualityTier.DATA_OK
                else 0.5
            )
            confidence = ConfidenceVector(
                raw_score=float(raw_score),
                model_score=None,
                calibrated_probability=None,
                uncertainty=None,
                quality_score=quality_score,
                calibration_state=CalibrationState.NOT_CALIBRATED,
            )

            det_prov = Provenance(
                provenance_id=f"prov-det-opt-{uuid4().hex[:12]}",
                source_asset_id=cand.provenance.source_asset_id,
                acquisition_metadata_ref=cand.provenance.acquisition_metadata_ref,
                pipeline_version="3.0.0",
                git_commit=cand.provenance.git_commit,
                config_hash=cand.provenance.config_hash,
                model_id=None,
                model_version=None,
                input_checksum=cand.provenance.output_checksum,
                output_checksum=hashlib.sha256(
                    cand.candidate_id.encode("utf-8")
                ).hexdigest()[:16],
                processing_run_id=cand.provenance.processing_run_id,
                processing_time_utc=datetime.now(timezone.utc),
                parent_provenance_id=cand.provenance.provenance_id,
            )

            aabb = (
                cand.bounding_box
                if getattr(cand, "bounding_box", None) is not None
                else AxisAlignedBoundingBox.from_tuple(cand.geom_px)
            )

            det = Detection(
                detection_id=f"det-opt-{uuid4().hex[:12]}",
                candidate_id=cand.candidate_id,
                sensor=SensorType.VHR_OPTICAL,
                geom_wgs84=cand.geom_wgs84,
                geom_px=aabb.to_tuple(),
                crs="EPSG:4326",
                target_class=TargetClass.AIRCRAFT,
                confidence=confidence,
                attributes=(
                    ("length_m", float(length_m)),
                    ("width_m", float(width_m)),
                    ("orientation_deg", float(orientation_deg)),
                    ("aspect_ratio", float(aspect_ratio)),
                    ("geometry_type", "AABB"),
                ),
                acquisition_time_utc=acquisition_time_utc
                or datetime.now(timezone.utc),
                provenance=det_prov,
                bounding_box=aabb,
            )
            detections.append(det)

        return tuple(detections)


HeuristicOpticalAircraftDetector = OpticalAircraftDetector
YoloObbAircraftDetector = OpticalModelDetector

__all__ = [
    "DetectionResult",
    "HeuristicOpticalAircraftDetector",
    "OpticalAircraftDetector",
    "OpticalModelDetector",
    "OpticalModelUnavailableError",
    "SkippedLowQuality",
    "YoloObbAircraftDetector",
]
