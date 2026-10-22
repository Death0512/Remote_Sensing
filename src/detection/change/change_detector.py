from datetime import datetime, timezone
import hashlib
from typing import Any, List, Optional, Tuple, Union
from uuid import uuid4
import numpy as np
import scipy.ndimage
from shapely.geometry import Point, Polygon

from src.domain.candidate import Candidate
from src.domain.confidence import ConfidenceVector
from src.domain.detection import Detection
from src.domain.enums import (
    CalibrationState,
    Capability,
    QualityTier,
    SensorType,
    TargetClass,
)
from src.domain.provenance import Provenance
from src.preprocessing.registration.bitemporal import RegisteredPair


class SkippedLowQuality(tuple):
    def __new__(cls, reason: str = "QualityTier.GEOREGISTRATION_UNCERTAIN"):
        instance = super().__new__(cls, ())
        object.__setattr__(instance, "reason", reason)
        return instance

    def __init__(self, reason: str = "QualityTier.GEOREGISTRATION_UNCERTAIN") -> None:
        self.reason = reason


class BitemporalChangeDetector:
    capability: Capability = Capability.IMPLEMENTED

    def __init__(
        self,
        diff_threshold: float = 0.3,
        min_area_px: int = 5,
        max_area_px: int = 100000,
        morph_kernel_size: int = 3,
    ) -> None:
        self.diff_threshold = diff_threshold
        self.min_area_px = min_area_px
        self.max_area_px = max_area_px
        self.morph_kernel_size = morph_kernel_size

    def _to_2d(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4):
                return np.mean(arr, axis=0)
            return np.mean(arr, axis=2)
        return arr

    def detect(
        self,
        registered_pair: RegisteredPair,
        **kwargs: Any,
    ) -> Tuple[Detection, ...]:
        if registered_pair.quality_tier == QualityTier.GEOREGISTRATION_UNCERTAIN:
            return SkippedLowQuality("QualityTier.GEOREGISTRATION_UNCERTAIN")

        if registered_pair.quality_tier not in (QualityTier.DATA_OK, QualityTier.DATA_PARTIAL):
            tier_val = (
                registered_pair.quality_tier.value
                if hasattr(registered_pair.quality_tier, "value")
                else str(registered_pair.quality_tier)
            )
            return SkippedLowQuality(f"QualityTier.{tier_val}")

        a_2d = self._to_2d(registered_pair.aligned_arr_a).astype(np.float32)
        b_2d = self._to_2d(registered_pair.aligned_arr_b).astype(np.float32)

        abs_diff = np.abs(b_2d - a_2d)
        denom = np.maximum(np.abs(a_2d) + np.abs(b_2d), 1e-6)
        norm_diff = abs_diff / denom

        change_mask = (abs_diff > self.diff_threshold) | (
            (norm_diff > self.diff_threshold) & (abs_diff > (self.diff_threshold * 0.2))
        )

        if self.morph_kernel_size > 1:
            structure = np.ones(
                (self.morph_kernel_size, self.morph_kernel_size), dtype=bool
            )
            filtered_mask = scipy.ndimage.binary_opening(change_mask, structure=structure)
        else:
            filtered_mask = change_mask

        labeled, num_features = scipy.ndimage.label(filtered_mask)
        if num_features == 0:
            return ()

        h, w = a_2d.shape
        obs_b = registered_pair.obs_b
        has_bounds = (
            obs_b.bounds_wgs84 is not None
            and not obs_b.bounds_wgs84.is_empty
        )
        if has_bounds:
            minx, miny, maxx, maxy = obs_b.bounds_wgs84.bounds
            valid_geom = maxx > minx and maxy > miny and h > 0 and w > 0
        else:
            valid_geom = False

        quality_score = (
            1.0
            if registered_pair.quality_tier == QualityTier.DATA_OK
            else 0.5
        )

        detections: List[Detection] = []
        for lbl in range(1, num_features + 1):
            coords = np.argwhere(labeled == lbl)
            area_px = len(coords)
            if area_px < self.min_area_px or area_px > self.max_area_px:
                continue

            min_row = int(np.min(coords[:, 0]))
            max_row = int(np.max(coords[:, 0]))
            min_col = int(np.min(coords[:, 1]))
            max_col = int(np.max(coords[:, 1]))
            geom_px = (float(min_col), float(min_row), float(max_col), float(max_row))

            c_row = float(np.mean(coords[:, 0]))
            c_col = float(np.mean(coords[:, 1]))

            diff_vals = abs_diff[coords[:, 0], coords[:, 1]]
            norm_vals = norm_diff[coords[:, 0], coords[:, 1]]
            mean_change = float(np.mean(norm_vals))
            peak_change = float(np.max(norm_vals))
            mean_abs_diff = float(np.mean(diff_vals))

            if valid_geom:
                lon_min = minx + (min_col / float(w)) * (maxx - minx)
                lon_max = minx + (max_col / float(w)) * (maxx - minx)
                lat_max = maxy - (min_row / float(h)) * (maxy - miny)
                lat_min = maxy - (max_row / float(h)) * (maxy - miny)
                geom_wgs84 = Polygon(
                    [
                        (lon_min, lat_min),
                        (lon_max, lat_min),
                        (lon_max, lat_max),
                        (lon_min, lat_max),
                        (lon_min, lat_min),
                    ]
                )
            else:
                geom_wgs84 = Point(c_col, c_row)

            cand_id = f"cand-chg-{uuid4().hex[:12]}"
            cand = Candidate(
                candidate_id=cand_id,
                processed_id=f"{registered_pair.obs_a.observation_id}:{obs_b.observation_id}",
                geom_px=geom_px,
                geom_wgs84=geom_wgs84,
                generator="BitemporalChangeDetector",
                generator_capability=self.capability,
                features=(
                    ("change_magnitude", mean_change),
                    ("peak_change", peak_change),
                    ("mean_abs_diff", mean_abs_diff),
                    ("area_px", float(area_px)),
                    ("centroid_col", c_col),
                    ("centroid_row", c_row),
                ),
                quality_flags=tuple(obs_b.quality_flags),
                provenance=registered_pair.provenance,
            )

            confidence = ConfidenceVector(
                raw_score=min(1.0, max(0.0, mean_change)),
                model_score=None,
                calibrated_probability=None,
                uncertainty=None,
                quality_score=quality_score,
                calibration_state=CalibrationState.NOT_CALIBRATED,
            )

            det_prov = Provenance(
                provenance_id=f"prov-det-chg-{uuid4().hex[:12]}",
                source_asset_id=obs_b.observation_id,
                acquisition_metadata_ref=obs_b.metadata_uri,
                pipeline_version="2.0.0",
                git_commit=registered_pair.provenance.git_commit,
                config_hash=registered_pair.provenance.config_hash,
                model_id=None,
                model_version=None,
                input_checksum=registered_pair.provenance.output_checksum,
                output_checksum=hashlib.sha256(
                    f"{cand_id}:{mean_change:.4f}".encode("utf-8")
                ).hexdigest()[:16],
                processing_run_id=registered_pair.provenance.processing_run_id,
                processing_time_utc=datetime.now(timezone.utc),
                parent_provenance_id=registered_pair.provenance.provenance_id,
            )

            det = Detection(
                detection_id=f"det-chg-{uuid4().hex[:12]}",
                candidate_id=cand.candidate_id,
                sensor=obs_b.sensor,
                geom_wgs84=geom_wgs84,
                geom_px=geom_px,
                crs=obs_b.crs,
                target_class=TargetClass.INFRASTRUCTURE_CHANGE,
                confidence=confidence,
                attributes=(
                    ("change_magnitude", mean_change),
                    ("peak_change", peak_change),
                    ("mean_abs_diff", mean_abs_diff),
                    ("area_px", float(area_px)),
                    ("residual_px", registered_pair.residual_px),
                ),
                acquisition_time_utc=obs_b.acquisition_time_utc,
                provenance=det_prov,
            )
            detections.append(det)

        return tuple(detections)
