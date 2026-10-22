from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from typing import Optional
from uuid import uuid4
import cv2
import numpy as np
import scipy.ndimage

from src.domain.enums import Capability, QualityTier, SensorType
from src.domain.observation import Observation
from src.domain.provenance import Provenance


@dataclass(frozen=True)
class RegisteredPair:
    obs_a: Observation
    obs_b: Observation
    aligned_arr_a: np.ndarray
    aligned_arr_b: np.ndarray
    residual_px: float
    quality_tier: QualityTier
    provenance: Provenance

    def __post_init__(self) -> None:
        if isinstance(self.quality_tier, str):
            object.__setattr__(self, "quality_tier", QualityTier(self.quality_tier))


class BitemporalRegistration:
    capability: Capability = Capability.IMPLEMENTED

    def __init__(self, residual_threshold_px: float = 0.5) -> None:
        self.residual_threshold_px = residual_threshold_px

    def _to_2d(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4):
                return np.mean(arr, axis=0)
            return np.mean(arr, axis=2)
        return arr

    def align(
        self,
        obs_a: Observation,
        obs_b: Observation,
        arr_a: np.ndarray,
        arr_b: np.ndarray,
    ) -> RegisteredPair:
        if obs_a.sensor != obs_b.sensor:
            raise ValueError("Incompatible observation sensors for bitemporal registration")
        if obs_a.product != obs_b.product:
            raise ValueError("Incompatible observation products for bitemporal registration")
        if abs(obs_a.ground_sample_distance_m - obs_b.ground_sample_distance_m) > 1e-3:
            raise ValueError("Incompatible observation resolutions for bitemporal registration")
        if arr_a.shape != arr_b.shape:
            raise ValueError("Incompatible array shapes for bitemporal registration")

        a_2d = self._to_2d(arr_a).astype(np.float32)
        b_2d = self._to_2d(arr_b).astype(np.float32)

        std_a = float(np.std(a_2d))
        std_b = float(np.std(b_2d))

        if std_a < 1e-5 or std_b < 1e-5:
            shift_x, shift_y = 0.0, 0.0
            residual_px = 0.0
        else:
            shift, _ = cv2.phaseCorrelate(a_2d, b_2d)
            shift_x, shift_y = float(shift[0]), float(shift[1])
            if np.isnan(shift_x) or np.isnan(shift_y):
                shift_x, shift_y = 0.0, 0.0
            residual_px = float(np.hypot(shift_x, shift_y))

        if residual_px > self.residual_threshold_px:
            quality_tier = QualityTier.GEOREGISTRATION_UNCERTAIN
            aligned_arr_a = arr_a.copy()
            aligned_arr_b = arr_b.copy()
        else:
            quality_tier = QualityTier.DATA_OK
            aligned_arr_a = arr_a.copy()
            if abs(shift_x) > 1e-4 or abs(shift_y) > 1e-4:
                if arr_b.ndim == 3:
                    aligned_arr_b = scipy.ndimage.shift(
                        arr_b, (0, -shift_y, -shift_x), order=1, mode="nearest"
                    )
                else:
                    aligned_arr_b = scipy.ndimage.shift(
                        arr_b, (-shift_y, -shift_x), order=1, mode="nearest"
                    )
            else:
                aligned_arr_b = arr_b.copy()

        prov = Provenance(
            provenance_id=f"prov-reg-{uuid4().hex[:12]}",
            source_asset_id=f"{obs_a.observation_id}+{obs_b.observation_id}",
            acquisition_metadata_ref=f"{obs_a.metadata_uri}+{obs_b.metadata_uri}",
            pipeline_version="2.0.0",
            git_commit=obs_a.provenance.git_commit,
            config_hash=hashlib.sha256(
                f"residual_threshold:{self.residual_threshold_px}".encode("utf-8")
            ).hexdigest()[:16],
            model_id=None,
            model_version=None,
            input_checksum=hashlib.sha256(
                f"{obs_a.provenance.output_checksum}:{obs_b.provenance.output_checksum}".encode(
                    "utf-8"
                )
            ).hexdigest()[:16],
            output_checksum=hashlib.sha256(
                f"{residual_px:.4f}:{quality_tier.value}".encode("utf-8")
            ).hexdigest()[:16],
            processing_run_id=obs_a.provenance.processing_run_id,
            processing_time_utc=datetime.now(timezone.utc),
            parent_provenance_id=obs_a.provenance.provenance_id,
        )

        return RegisteredPair(
            obs_a=obs_a,
            obs_b=obs_b,
            aligned_arr_a=aligned_arr_a,
            aligned_arr_b=aligned_arr_b,
            residual_px=residual_px,
            quality_tier=quality_tier,
            provenance=prov,
        )
