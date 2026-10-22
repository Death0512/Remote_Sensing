from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
from uuid import uuid4
import numpy as np
from shapely.geometry import Point

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
from src.domain.observation import ProcessedObservation
from src.domain.provenance import Provenance
from src.ingestion.providers.base import ProviderNotAvailableError


class SkippedLowQuality(tuple):
    def __new__(cls, reason: str = "QualityTier.CLOUD_CONTAMINATED"):
        instance = super().__new__(cls, ())
        object.__setattr__(instance, "reason", reason)
        return instance

    def __init__(self, reason: str = "QualityTier.CLOUD_CONTAMINATED") -> None:
        self.reason = reason


DetectionResult = Union[Tuple[Detection, ...], SkippedLowQuality]


class OpticalModelDetector:
    capability: Capability = Capability.EXTERNAL_MODEL

    def __init__(
        self,
        weights_path: Union[str, Path],
        conf_threshold: float = 0.25,
    ) -> None:
        path = Path(weights_path)
        if not path.is_file():
            raise ProviderNotAvailableError(
                f"Model checkpoint not found: {weights_path}"
            )

        try:
            from ultralytics import YOLO

            self.model = YOLO(str(path))
        except Exception as exc:
            raise ProviderNotAvailableError(
                f"Failed to load model from {weights_path}: {exc}"
            ) from exc

        self.weights_path = path
        self.conf_threshold = conf_threshold

    def detect(
        self,
        processed_observation: ProcessedObservation,
        image: np.ndarray,
        pixel_spacing_m: float = 0.5,
        affine_transform: Optional[Any] = None,
        acquisition_time_utc: Optional[datetime] = None,
    ) -> DetectionResult:
        if (
            processed_observation.quality_tier
            == QualityTier.CLOUD_CONTAMINATED
        ):
            return SkippedLowQuality("QualityTier.CLOUD_CONTAMINATED")

        results = self.model(image, conf=self.conf_threshold)
        detections: List[Detection] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                score = float(box.conf[0].cpu().numpy())
                min_col, min_row, max_col, max_row = (
                    float(xyxy[0]),
                    float(xyxy[1]),
                    float(xyxy[2]),
                    float(xyxy[3]),
                )
                c_col = (min_col + max_col) / 2.0
                c_row = (min_row + max_row) / 2.0
                if affine_transform is not None and hasattr(
                    affine_transform, "pixel_to_geo"
                ):
                    geo_x, geo_y = affine_transform.pixel_to_geo(c_col, c_row)
                    geom_wgs84 = Point(geo_x, geo_y)
                else:
                    geom_wgs84 = Point(c_col, c_row)

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
                    pipeline_version="2.0.0",
                    git_commit=processed_observation.provenance.git_commit,
                    config_hash=processed_observation.provenance.config_hash,
                    model_id=str(self.weights_path.name),
                    model_version="1.0",
                    input_checksum=processed_observation.provenance.output_checksum,
                    output_checksum=hashlib.sha256(
                        f"{c_col}_{c_row}_{score}".encode("utf-8")
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
                    geom_px=(min_col, min_row, max_col, max_row),
                    crs="EPSG:4326",
                    target_class=TargetClass.AIRCRAFT,
                    confidence=conf,
                    attributes=(("score", score),),
                    acquisition_time_utc=acquisition_time_utc
                    or datetime.now(timezone.utc),
                    provenance=det_prov,
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
        self.weights_path = Path(weights_path) if weights_path else None
        self.min_length_m = min_length_m
        self.max_length_m = max_length_m
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.pixel_spacing_m = pixel_spacing_m

        self.model = None
        if self.weights_path and self.weights_path.is_file():
            try:
                from ultralytics import YOLO

                self.model = YOLO(str(self.weights_path))
                self.capability = Capability.EXTERNAL_MODEL
            except Exception:
                self.model = None
                self.capability = Capability.HEURISTIC
        else:
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
                pipeline_version="2.0.0",
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

            det = Detection(
                detection_id=f"det-opt-{uuid4().hex[:12]}",
                candidate_id=cand.candidate_id,
                sensor=SensorType.VHR_OPTICAL,
                geom_wgs84=cand.geom_wgs84,
                geom_px=cand.geom_px,
                crs="EPSG:4326",
                target_class=TargetClass.AIRCRAFT,
                confidence=confidence,
                attributes=(
                    ("length_m", float(length_m)),
                    ("width_m", float(width_m)),
                    ("orientation_deg", float(orientation_deg)),
                    ("aspect_ratio", float(aspect_ratio)),
                ),
                acquisition_time_utc=acquisition_time_utc
                or datetime.now(timezone.utc),
                provenance=det_prov,
            )
            detections.append(det)

        return tuple(detections)
