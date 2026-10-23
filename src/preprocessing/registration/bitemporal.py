from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
from typing import Optional
from uuid import uuid4
import cv2
import numpy as np
import scipy.ndimage

from src.domain.enums import Capability, QualityTier, SensorType
from src.domain.observation import Observation
from src.domain.provenance import Provenance


@dataclass(frozen=True)
class RegistrationResult:
    success: bool
    estimated_displacement_px: tuple[float, float]
    residual_rmse_px: float
    residual_median_px: float
    residual_p95_px: float
    quality_tier: QualityTier
    aligned_pair: tuple[np.ndarray, np.ndarray]
    transform_matrix: Optional[tuple[float, ...]] = None
    failure_reason: Optional[str] = None
    obs_a: Optional[Observation] = None
    obs_b: Optional[Observation] = None
    provenance: Optional[Provenance] = None
    schema_version: str = "registration_result.v3"

    def __post_init__(self) -> None:
        if isinstance(self.quality_tier, str):
            object.__setattr__(self, "quality_tier", QualityTier(self.quality_tier))
        if not isinstance(self.estimated_displacement_px, tuple):
            object.__setattr__(
                self, "estimated_displacement_px", tuple(self.estimated_displacement_px)
            )
        if not isinstance(self.aligned_pair, tuple):
            object.__setattr__(self, "aligned_pair", tuple(self.aligned_pair))
        if self.transform_matrix is not None and not isinstance(
            self.transform_matrix, tuple
        ):
            object.__setattr__(self, "transform_matrix", tuple(self.transform_matrix))

    @property
    def residual_px(self) -> float:
        return self.residual_rmse_px

    @property
    def aligned_arr_a(self) -> np.ndarray:
        return self.aligned_pair[0]

    @property
    def aligned_arr_b(self) -> np.ndarray:
        return self.aligned_pair[1]


RegisteredPair = RegistrationResult


class BitemporalRegistration:
    capability: Capability = Capability.IMPLEMENTED

    def __init__(
        self,
        max_residual_rmse_px: float = 0.75,
        max_displacement_px: Optional[float] = None,
        residual_threshold_px: Optional[float] = None,
    ) -> None:
        if residual_threshold_px is not None:
            self.max_residual_rmse_px = float(residual_threshold_px)
            self.max_displacement_px = float(residual_threshold_px)
            self.residual_threshold_px = float(residual_threshold_px)
        else:
            self.max_residual_rmse_px = float(max_residual_rmse_px)
            self.max_displacement_px = (
                float(max_displacement_px) if max_displacement_px is not None else None
            )
            self.residual_threshold_px = float(max_residual_rmse_px)

    def _to_2d(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4):
                return np.mean(arr, axis=0)
            return np.mean(arr, axis=2)
        return arr

    def _compute_post_warp_residuals(
        self, a_2d: np.ndarray, aligned_b_2d: np.ndarray
    ) -> tuple[float, float, float]:
        min_val = min(float(np.min(a_2d)), float(np.min(aligned_b_2d)))
        max_val = max(float(np.max(a_2d)), float(np.max(aligned_b_2d)))
        rng = max_val - min_val
        if rng > 1e-5:
            norm_a = np.clip((a_2d - min_val) / rng * 255.0, 0, 255).astype(np.uint8)
            norm_b = np.clip((aligned_b_2d - min_val) / rng * 255.0, 0, 255).astype(
                np.uint8
            )
            pts = cv2.goodFeaturesToTrack(
                norm_a, maxCorners=100, qualityLevel=0.05, minDistance=5
            )
            if pts is not None and len(pts) >= 4:
                pts_b, sf, _ = cv2.calcOpticalFlowPyrLK(norm_a, norm_b, pts, None)
                if pts_b is not None and sf is not None:
                    pts_back, sb, _ = cv2.calcOpticalFlowPyrLK(
                        norm_b, norm_a, pts_b, None
                    )
                    if pts_back is not None and sb is not None:
                        fb_err = np.linalg.norm(
                            pts.reshape(-1, 2) - pts_back.reshape(-1, 2), axis=1
                        )
                        valid = (sf.ravel() == 1) & (sb.ravel() == 1) & (fb_err < 0.5)
                        if np.sum(valid) >= 4:
                            p_a = pts[valid].reshape(-1, 2)
                            p_b = pts_b[valid].reshape(-1, 2)
                            res = np.sqrt(np.sum((p_a - p_b) ** 2, axis=1))
                            rmse = float(np.sqrt(np.mean(res**2)))
                            median = float(np.median(res))
                            p95 = float(np.percentile(res, 95))
                            return rmse, median, p95

        h, w = a_2d.shape
        grid_n = 4
        patch_h = h // grid_n
        patch_w = w // grid_n
        patch_residuals = []
        if patch_h >= 10 and patch_w >= 10:
            for gi in range(grid_n):
                for gj in range(grid_n):
                    pa = a_2d[
                        gi * patch_h : (gi + 1) * patch_h,
                        gj * patch_w : (gj + 1) * patch_w,
                    ]
                    pb = aligned_b_2d[
                        gi * patch_h : (gi + 1) * patch_h,
                        gj * patch_w : (gj + 1) * patch_w,
                    ]
                    if float(np.std(pa)) > 1e-4 and float(np.std(pb)) > 1e-4:
                        sh, resp = cv2.phaseCorrelate(pa, pb)
                        if resp >= 0.15 and not (np.isnan(sh[0]) or np.isnan(sh[1])):
                            patch_residuals.append(float(np.hypot(sh[0], sh[1])))

        if len(patch_residuals) >= 2:
            arr_res = np.array(patch_residuals, dtype=np.float64)
            rmse = float(np.sqrt(np.mean(arr_res**2)))
            median = float(np.median(arr_res))
            p95 = float(np.percentile(arr_res, 95))
            return rmse, median, p95

        if float(np.std(a_2d)) > 1e-5 and float(np.std(aligned_b_2d)) > 1e-5:
            sh, _ = cv2.phaseCorrelate(a_2d, aligned_b_2d)
            sx, sy = float(sh[0]), float(sh[1])
            if np.isnan(sx) or np.isnan(sy):
                sx, sy = 0.0, 0.0
            res = float(np.hypot(sx, sy))
        else:
            res = 0.0
        return res, res, res

    def _create_provenance(
        self,
        obs_a: Observation,
        obs_b: Observation,
        residual_rmse_px: float,
        quality_tier: QualityTier,
    ) -> Provenance:
        git_commit = (
            obs_a.provenance.git_commit
            if obs_a.provenance
            else (obs_b.provenance.git_commit if obs_b.provenance else "unknown")
        )
        run_id = (
            obs_a.provenance.processing_run_id
            if obs_a.provenance
            else (
                obs_b.provenance.processing_run_id if obs_b.provenance else "unknown"
            )
        )
        parent_id = obs_a.provenance.provenance_id if obs_a.provenance else None
        in_chk_a = obs_a.provenance.output_checksum if obs_a.provenance else "none_a"
        in_chk_b = obs_b.provenance.output_checksum if obs_b.provenance else "none_b"
        pipeline_version = (
            obs_a.provenance.pipeline_version
            if (obs_a.provenance and obs_a.provenance.pipeline_version)
            else (
                obs_b.provenance.pipeline_version
                if (obs_b.provenance and obs_b.provenance.pipeline_version)
                else "3.0.0"
            )
        )
        return Provenance(
            provenance_id=f"prov-reg-{uuid4().hex[:12]}",
            source_asset_id=f"{obs_a.observation_id}+{obs_b.observation_id}",
            acquisition_metadata_ref=f"{obs_a.metadata_uri}+{obs_b.metadata_uri}",
            pipeline_version=pipeline_version,
            git_commit=git_commit,
            config_hash=hashlib.sha256(
                f"max_residual_rmse:{self.max_residual_rmse_px}".encode("utf-8")
            ).hexdigest()[:16],
            model_id=None,
            model_version=None,
            input_checksum=hashlib.sha256(
                f"{in_chk_a}:{in_chk_b}".encode("utf-8")
            ).hexdigest()[:16],
            output_checksum=hashlib.sha256(
                f"{residual_rmse_px:.4f}:{quality_tier.value}".encode("utf-8")
            ).hexdigest()[:16],
            processing_run_id=run_id,
            processing_time_utc=datetime.now(timezone.utc),
            parent_provenance_id=parent_id,
        )

    def align(
        self,
        obs_a: Observation,
        obs_b: Observation,
        arr_a: np.ndarray,
        arr_b: np.ndarray,
    ) -> RegistrationResult:
        if obs_a.sensor != obs_b.sensor:
            raise ValueError(
                f"Incompatible observation sensors for bitemporal registration: {obs_a.sensor} != {obs_b.sensor}"
            )
        if obs_a.product != obs_b.product:
            raise ValueError(
                "Incompatible observation products for bitemporal registration"
            )
        if obs_a.crs != obs_b.crs:
            raise ValueError(
                f"Incompatible observation CRS for bitemporal registration: {obs_a.crs} != {obs_b.crs}"
            )
        if abs(obs_a.ground_sample_distance_m - obs_b.ground_sample_distance_m) > 1e-3:
            raise ValueError(
                "Incompatible observation resolutions for bitemporal registration"
            )
        if arr_a.shape != arr_b.shape:
            raise ValueError("Incompatible array shapes for bitemporal registration")

        if (
            obs_a.bounds_wgs84 is None
            or obs_b.bounds_wgs84 is None
            or getattr(obs_a.bounds_wgs84, "is_empty", False)
            or getattr(obs_b.bounds_wgs84, "is_empty", False)
            or obs_a.ground_sample_distance_m <= 0.0
            or obs_b.ground_sample_distance_m <= 0.0
        ):
            prov = self._create_provenance(
                obs_a, obs_b, float("inf"), QualityTier.CORRUPTED_METADATA
            )
            return RegistrationResult(
                success=False,
                estimated_displacement_px=(0.0, 0.0),
                residual_rmse_px=float("inf"),
                residual_median_px=float("inf"),
                residual_p95_px=float("inf"),
                quality_tier=QualityTier.CORRUPTED_METADATA,
                aligned_pair=(arr_a.copy(), arr_b.copy()),
                transform_matrix=None,
                failure_reason="Corrupted metadata in observation pairs",
                obs_a=obs_a,
                obs_b=obs_b,
                provenance=prov,
            )

        a_2d = self._to_2d(arr_a).astype(np.float32)
        b_2d = self._to_2d(arr_b).astype(np.float32)

        std_a = float(np.std(a_2d))
        std_b = float(np.std(b_2d))

        if std_a < 1e-5 or std_b < 1e-5:
            shift_x, shift_y = 0.0, 0.0
        else:
            shift, _ = cv2.phaseCorrelate(a_2d, b_2d)
            shift_x, shift_y = float(shift[0]), float(shift[1])
            if np.isnan(shift_x) or np.isnan(shift_y):
                shift_x, shift_y = 0.0, 0.0

        estimated_disp = (shift_x, shift_y)
        disp_mag = float(np.hypot(shift_x, shift_y))

        if self.max_displacement_px is not None and disp_mag > self.max_displacement_px:
            prov = self._create_provenance(
                obs_a, obs_b, disp_mag, QualityTier.GEOREGISTRATION_UNCERTAIN
            )
            return RegistrationResult(
                success=False,
                estimated_displacement_px=estimated_disp,
                residual_rmse_px=disp_mag,
                residual_median_px=disp_mag,
                residual_p95_px=disp_mag,
                quality_tier=QualityTier.GEOREGISTRATION_UNCERTAIN,
                aligned_pair=(arr_a.copy(), arr_b.copy()),
                transform_matrix=None,
                failure_reason=f"Estimated displacement ({disp_mag:.2f} px) exceeds limit ({self.max_displacement_px:.2f} px)",
                obs_a=obs_a,
                obs_b=obs_b,
                provenance=prov,
            )

        transform_matrix = (1.0, 0.0, -shift_x, 0.0, 1.0, -shift_y)
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

        aligned_arr_a = arr_a.copy()
        aligned_b_2d = self._to_2d(aligned_arr_b).astype(np.float32)

        residual_rmse_px, residual_median_px, residual_p95_px = (
            self._compute_post_warp_residuals(a_2d, aligned_b_2d)
        )

        if residual_rmse_px > self.max_residual_rmse_px:
            success = False
            quality_tier = QualityTier.GEOREGISTRATION_UNCERTAIN
            failure_reason = f"Residual RMSE ({residual_rmse_px:.4f} px) exceeds threshold ({self.max_residual_rmse_px:.4f} px)"
        else:
            success = True
            quality_tier = QualityTier.DATA_OK
            failure_reason = None

        prov = self._create_provenance(
            obs_a, obs_b, residual_rmse_px, quality_tier
        )

        return RegistrationResult(
            success=success,
            estimated_displacement_px=estimated_disp,
            residual_rmse_px=residual_rmse_px,
            residual_median_px=residual_median_px,
            residual_p95_px=residual_p95_px,
            quality_tier=quality_tier,
            aligned_pair=(aligned_arr_a, aligned_arr_b),
            transform_matrix=transform_matrix,
            failure_reason=failure_reason,
            obs_a=obs_a,
            obs_b=obs_b,
            provenance=prov,
        )
