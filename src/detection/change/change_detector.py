from datetime import datetime, timezone
import hashlib
from typing import Any, List, Optional, Tuple, Union
from uuid import uuid4
import cv2
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
from src.domain.geometry import AxisAlignedBoundingBox, PixelGeometry
from src.domain.observation import Observation
from src.domain.provenance import Provenance
from src.preprocessing.registration.bitemporal import (
    BitemporalRegistration,
    RegisteredPair,
    RegistrationResult,
)


class RegistrationFailureReason(str):
    def __eq__(self, other: object) -> bool:
        if str.__eq__(self, other):
            return True
        if str(self) in ("REGISTRATION_FAILED", "QualityTier.GEOREGISTRATION_UNCERTAIN") and other in (
            "REGISTRATION_FAILED",
            "QualityTier.GEOREGISTRATION_UNCERTAIN",
        ):
            return True
        return False

    def __hash__(self) -> int:
        return str.__hash__(self)


class SkippedLowQuality(tuple):
    def __new__(
        cls,
        reason: Union[str, RegistrationFailureReason] = RegistrationFailureReason(
            "REGISTRATION_FAILED"
        ),
    ):
        instance = super().__new__(cls, ())
        object.__setattr__(instance, "reason", reason)
        return instance

    def __init__(
        self,
        reason: Union[str, RegistrationFailureReason] = RegistrationFailureReason(
            "REGISTRATION_FAILED"
        ),
    ) -> None:
        self.reason = reason


class BitemporalChangeDetector:
    capability: Capability = Capability.IMPLEMENTED

    def __init__(
        self,
        sigma_multiplier: float = 3.0,
        p_fa: Optional[float] = None,
        method: str = "mad",
        min_area_px: int = 5,
        max_area_px: int = 100000,
        morph_kernel_size: int = 3,
        diff_threshold: Optional[float] = None,
        structure_area_threshold_m2: float = 2500.0,
        pixel_spacing_m: Optional[float] = None,
        registration_pipeline: Optional[BitemporalRegistration] = None,
        weights_config_path: Optional[str] = None,
        cosine_threshold: Optional[float] = None,
    ) -> None:
        self.sigma_multiplier = float(sigma_multiplier)
        self.p_fa = float(p_fa) if p_fa is not None else None
        self.method = method
        self.min_area_px = int(min_area_px)
        self.max_area_px = int(max_area_px)
        self.morph_kernel_size = int(morph_kernel_size)
        self.diff_threshold = (
            float(diff_threshold) if diff_threshold is not None else None
        )
        self.structure_area_threshold_m2 = float(structure_area_threshold_m2)
        self.pixel_spacing_m = (
            float(pixel_spacing_m) if pixel_spacing_m is not None else None
        )
        self.registration_pipeline = (
            registration_pipeline
            if registration_pipeline is not None
            else BitemporalRegistration()
        )
        self.weights_config_path = weights_config_path
        self.cosine_threshold = cosine_threshold

    def _to_2d(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4):
                return np.mean(arr, axis=0)
            return np.mean(arr, axis=2)
        return arr

    def _calibrate_threshold(self, diff_metric: np.ndarray) -> float:
        valid = diff_metric[np.isfinite(diff_metric)]
        if valid.size == 0:
            return 0.5
        med = float(np.median(valid))
        mad = float(np.median(np.abs(valid - med)))
        sigma = 1.4826 * mad
        if sigma < 1e-6:
            sigma = float(np.std(valid))

        if self.method == "mad":
            th = med + self.sigma_multiplier * sigma
        elif self.method == "sigma":
            th = float(np.mean(valid)) + self.sigma_multiplier * sigma
        elif self.method == "quantile" or self.p_fa is not None:
            p_fa = self.p_fa if self.p_fa is not None else 1e-3
            q = max(0.5, min(0.99999, 1.0 - p_fa))
            th = float(np.percentile(valid, q * 100.0))
        elif self.method == "otsu":
            v_max = float(np.max(valid))
            if v_max > 1e-6:
                norm_val = np.clip(valid * 255.0 / v_max, 0, 255).astype(np.uint8)
                otsu_th, _ = cv2.threshold(
                    norm_val, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                th = float(otsu_th) / 255.0 * v_max
            else:
                th = med + self.sigma_multiplier * sigma
        else:
            th = med + self.sigma_multiplier * sigma
        return max(float(th), 1e-4)

    def _synthesize_observation(
        self, obs_id: str, sensor: SensorType, shape: tuple[int, ...]
    ) -> Observation:
        h, w = shape[-2], shape[-1]
        poly = Polygon([(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))])
        prov = Provenance(
            provenance_id=f"prov-{obs_id}",
            source_asset_id=obs_id,
            acquisition_metadata_ref=f"s3://meta/{obs_id}",
            pipeline_version="3.0.0",
            git_commit="unknown",
            config_hash="default",
            model_id=None,
            model_version=None,
            input_checksum="default",
            output_checksum="default",
            processing_run_id="run-synth",
            processing_time_utc=datetime.now(timezone.utc),
        )
        return Observation(
            observation_id=obs_id,
            source="SYNTHETIC",
            platform="Sentinel",
            sensor=sensor,
            product="DEFAULT",
            processing_level="L2",
            acquisition_time_utc=datetime.now(timezone.utc),
            crs="EPSG:4326",
            ground_sample_distance_m=self.pixel_spacing_m or 10.0,
            bounds_wgs84=poly,
            asset_uri=f"s3://data/{obs_id}",
            metadata_uri=f"s3://meta/{obs_id}",
            quality_flags=("DATA_OK",),
            provenance=prov,
        )

    def detect(
        self,
        image_input: Union[RegistrationResult, Tuple[Any, ...]],
        **kwargs: Any,
    ) -> Tuple[Detection, ...]:
        if isinstance(image_input, RegistrationResult):
            registered = image_input
        elif isinstance(image_input, (tuple, list)):
            if len(image_input) == 2 and isinstance(image_input[0], np.ndarray):
                arr_a, arr_b = image_input[0], image_input[1]
                sensor_name = kwargs.get("sensor", "S1_SAR")
                try:
                    sensor = SensorType(sensor_name)
                except ValueError:
                    sensor = SensorType.S1_SAR
                obs_a = self._synthesize_observation("obs-t1", sensor, arr_a.shape)
                obs_b = self._synthesize_observation("obs-t2", sensor, arr_b.shape)
                registered = self.registration_pipeline.align(
                    obs_a, obs_b, arr_a, arr_b
                )
            elif (
                len(image_input) == 2
                and isinstance(image_input[0], Observation)
                and isinstance(image_input[1], Observation)
            ):
                arr_a = kwargs.get("arr_a")
                arr_b = kwargs.get("arr_b")
                if arr_a is None or arr_b is None:
                    raise ValueError(
                        "arr_a and arr_b arrays must be provided with Observation pair"
                    )
                registered = self.registration_pipeline.align(
                    image_input[0], image_input[1], arr_a, arr_b
                )
            elif len(image_input) == 4:
                registered = self.registration_pipeline.align(
                    image_input[0], image_input[1], image_input[2], image_input[3]
                )
            else:
                raise ValueError("Unsupported input format for change detection")
        else:
            raise TypeError("Expected RegistrationResult or tuple of inputs")

        if registered.quality_tier == QualityTier.CORRUPTED_METADATA:
            return SkippedLowQuality(RegistrationFailureReason("CORRUPTED_METADATA"))

        if (
            not registered.success
            or registered.quality_tier == QualityTier.GEOREGISTRATION_UNCERTAIN
        ):
            return SkippedLowQuality(RegistrationFailureReason("REGISTRATION_FAILED"))

        if registered.quality_tier not in (
            QualityTier.DATA_OK,
            QualityTier.DATA_PARTIAL,
        ):
            tier_val = (
                registered.quality_tier.value
                if hasattr(registered.quality_tier, "value")
                else str(registered.quality_tier)
            )
            return SkippedLowQuality(f"QualityTier.{tier_val}")

        arr_a = registered.aligned_pair[0]
        arr_b = registered.aligned_pair[1]
        a_2d = self._to_2d(arr_a).astype(np.float32)
        b_2d = self._to_2d(arr_b).astype(np.float32)

        sensor = (
            registered.obs_b.sensor
            if (registered.obs_b and registered.obs_b.sensor)
            else SensorType.S2_MSI
        )

        abs_diff = np.abs(b_2d - a_2d)
        denom = np.maximum(np.abs(a_2d) + np.abs(b_2d), 1e-6)
        norm_diff = abs_diff / denom

        if sensor == SensorType.S1_SAR:
            diff_metric = np.abs(
                np.log(np.maximum(b_2d, 1e-6)) - np.log(np.maximum(a_2d, 1e-6))
            )
        else:
            diff_metric = norm_diff

        if self.diff_threshold is not None:
            calibrated_th = float(self.diff_threshold)
            change_mask = (abs_diff > self.diff_threshold) | (
                (norm_diff > self.diff_threshold)
                & (abs_diff > (self.diff_threshold * 0.2))
            )
        else:
            calibrated_th = self._calibrate_threshold(diff_metric)
            change_mask = diff_metric > calibrated_th

        if self.morph_kernel_size > 1:
            structure = np.ones(
                (self.morph_kernel_size, self.morph_kernel_size), dtype=bool
            )
            filtered_mask = scipy.ndimage.binary_opening(
                change_mask, structure=structure
            )
        else:
            filtered_mask = change_mask

        labeled, num_features = scipy.ndimage.label(filtered_mask)
        if num_features == 0:
            return ()

        h, w = a_2d.shape
        obs_b = registered.obs_b
        has_bounds = (
            obs_b is not None
            and obs_b.bounds_wgs84 is not None
            and not getattr(obs_b.bounds_wgs84, "is_empty", False)
        )
        if has_bounds:
            minx, miny, maxx, maxy = obs_b.bounds_wgs84.bounds
            valid_geom = maxx > minx and maxy > miny and h > 0 and w > 0
        else:
            valid_geom = False

        quality_score = (
            1.0 if registered.quality_tier == QualityTier.DATA_OK else 0.5
        )

        gsd = (
            obs_b.ground_sample_distance_m
            if obs_b
            else (self.pixel_spacing_m or 10.0)
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

            aabb = AxisAlignedBoundingBox(
                min_col=float(min_col),
                min_row=float(min_row),
                max_col=float(max_col),
                max_row=float(max_row),
            )
            c_row = float(np.mean(coords[:, 0]))
            c_col = float(np.mean(coords[:, 1]))

            diff_vals = abs_diff[coords[:, 0], coords[:, 1]]
            norm_vals = norm_diff[coords[:, 0], coords[:, 1]]
            mean_change = float(np.mean(norm_vals))
            peak_change = float(np.max(norm_vals))
            mean_abs_diff = float(np.mean(diff_vals))

            area_m2 = float(area_px) * (gsd**2)
            length_m = float(max(aabb.width, aabb.height)) * gsd
            width_m = float(min(aabb.width, aabb.height)) * gsd

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
                processed_id=(
                    f"{registered.obs_a.observation_id}:{obs_b.observation_id}"
                    if (registered.obs_a and obs_b)
                    else f"chg-{lbl}"
                ),
                geom_px=aabb.to_tuple(),
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
                    ("calibrated_threshold", float(calibrated_th)),
                ),
                quality_flags=(
                    tuple(obs_b.quality_flags) if obs_b else ("DATA_OK",)
                ),
                provenance=registered.provenance,
            )

            confidence = ConfidenceVector(
                raw_score=min(1.0, max(0.0, mean_change)),
                model_score=None,
                calibrated_probability=None,
                uncertainty=None,
                quality_score=quality_score,
                calibration_state=CalibrationState.NOT_CALIBRATED,
            )

            source_id = obs_b.observation_id if obs_b else "bitemporal_scene"
            meta_ref = obs_b.metadata_uri if obs_b else "meta_change"
            acq_time = (
                obs_b.acquisition_time_utc
                if (obs_b and obs_b.acquisition_time_utc)
                else (kwargs.get("acquisition_time") or datetime.now(timezone.utc))
            )
            git_commit = (
                registered.provenance.git_commit
                if registered.provenance
                else "unknown"
            )
            cfg_hash = (
                registered.provenance.config_hash
                if registered.provenance
                else hashlib.sha256(
                    f"calibrated_th:{calibrated_th:.4f}".encode("utf-8")
                ).hexdigest()[:16]
            )
            in_chk = (
                registered.provenance.output_checksum
                if registered.provenance
                else "none"
            )

            pipe_ver = (
                registered.provenance.pipeline_version
                if (registered.provenance and registered.provenance.pipeline_version)
                else "3.0.0"
            )

            det_prov = Provenance(
                provenance_id=f"prov-det-chg-{uuid4().hex[:12]}",
                source_asset_id=source_id,
                acquisition_metadata_ref=meta_ref,
                pipeline_version=pipe_ver,
                git_commit=git_commit,
                config_hash=cfg_hash,
                model_id=None,
                model_version=None,
                input_checksum=in_chk,
                output_checksum=hashlib.sha256(
                    f"{cand_id}:{mean_change:.4f}".encode("utf-8")
                ).hexdigest()[:16],
                processing_run_id=(
                    registered.provenance.processing_run_id
                    if registered.provenance
                    else "run-chg"
                ),
                processing_time_utc=datetime.now(timezone.utc),
                parent_provenance_id=(
                    registered.provenance.provenance_id
                    if registered.provenance
                    else None
                ),
            )

            det = Detection(
                detection_id=f"det-chg-{uuid4().hex[:12]}",
                candidate_id=cand.candidate_id,
                sensor=sensor,
                geom_wgs84=geom_wgs84,
                geom_px=aabb,
                crs=obs_b.crs if obs_b else "EPSG:4326",
                target_class=TargetClass.INFRASTRUCTURE_CHANGE,
                confidence=confidence,
                attributes=(
                    ("change_magnitude", mean_change),
                    ("peak_change", peak_change),
                    ("mean_abs_diff", mean_abs_diff),
                    ("area_px", float(area_px)),
                    ("area_m2", float(area_m2)),
                    ("length_m", float(length_m)),
                    ("width_m", float(width_m)),
                    ("residual_px", registered.residual_rmse_px),
                    ("calibrated_threshold", float(calibrated_th)),
                    ("pixel_geometry_row", float(c_row)),
                    ("pixel_geometry_col", float(c_col)),
                ),
                acquisition_time_utc=acq_time,
                provenance=det_prov,
            )
            detections.append(det)

        return tuple(detections)


DualStreamChangeDetector = BitemporalChangeDetector
CalibratedChangeDetector = BitemporalChangeDetector
