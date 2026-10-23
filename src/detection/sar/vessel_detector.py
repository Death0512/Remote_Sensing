from datetime import datetime, timezone
import hashlib
from typing import Any, List, Optional, Tuple
from uuid import uuid4
import numpy as np

from src.detection.sar.local_cfar import LocalCfarDetector
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


class SarVesselDetector:
    capability: Capability = Capability.IMPLEMENTED

    def __init__(
        self,
        cfar_detector: Optional[LocalCfarDetector] = None,
        min_length_m: float = 10.0,
        max_length_m: float = 450.0,
        min_width_m: float = 2.0,
        max_width_m: float = 100.0,
        min_aspect_ratio: float = 1.0,
        max_aspect_ratio: float = 15.0,
        pixel_spacing_m: float = 10.0,
    ) -> None:
        self.cfar_detector = cfar_detector or LocalCfarDetector(gsd_m=pixel_spacing_m)
        self.min_length_m = min_length_m
        self.max_length_m = max_length_m
        self.min_width_m = min_width_m
        self.max_width_m = max_width_m
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.pixel_spacing_m = pixel_spacing_m

    def detect(
        self,
        intensity: Optional[Any] = None,
        processed_observation: Optional[ProcessedObservation] = None,
        land_mask: Optional[np.ndarray] = None,
        pixel_spacing_m: Optional[float] = None,
        affine_transform: Optional[Any] = None,
        acquisition_time_utc: Optional[datetime] = None,
        **kwargs: Any,
    ) -> Tuple[Detection, ...]:
        if isinstance(intensity, ProcessedObservation) and (
            processed_observation is None or isinstance(processed_observation, np.ndarray)
        ):
            intensity, processed_observation = processed_observation, intensity
        elif processed_observation is None and "processed_observation" in kwargs:
            processed_observation = kwargs.pop("processed_observation")

        if intensity is None and "intensity" in kwargs:
            intensity = kwargs.pop("intensity")

        if intensity is None:
            raise ValueError("intensity array is required")

        spacing = pixel_spacing_m if pixel_spacing_m is not None else self.pixel_spacing_m
        candidates = self.cfar_detector.detect(
            intensity=intensity,
            processed_observation=processed_observation,
            land_mask=land_mask,
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
            snr = feat_dict.get("snr", 1.0)

            if not (self.min_length_m <= length_m <= self.max_length_m):
                continue
            if not (self.min_width_m <= width_m <= self.max_width_m):
                continue
            if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                continue

            quality_score = (
                1.0
                if (processed_observation and processed_observation.quality_tier == QualityTier.DATA_OK)
                else (0.85 if processed_observation is None else 0.5)
            )
            confidence = ConfidenceVector(
                raw_score=snr,
                model_score=None,
                calibrated_probability=None,
                uncertainty=None,
                quality_score=quality_score,
                calibration_state=CalibrationState.NOT_CALIBRATED,
            )

            det_prov = Provenance(
                provenance_id=f"prov-det-{uuid4().hex[:12]}",
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

            det = Detection(
                detection_id=f"det-{uuid4().hex[:12]}",
                candidate_id=cand.candidate_id,
                sensor=SensorType.S1_SAR,
                geom_wgs84=cand.geom_wgs84,
                geom_px=cand.geom_px,
                crs="EPSG:4326",
                target_class=TargetClass.VESSEL,
                confidence=confidence,
                attributes=(
                    ("length_m", float(length_m)),
                    ("width_m", float(width_m)),
                    ("orientation_deg", float(orientation_deg)),
                    ("aspect_ratio", float(aspect_ratio)),
                ),
                acquisition_time_utc=acquisition_time_utc or datetime.now(timezone.utc),
                provenance=det_prov,
            )
            detections.append(det)

        return tuple(detections)
