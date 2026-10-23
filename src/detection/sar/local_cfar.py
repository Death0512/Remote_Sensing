from datetime import datetime, timezone
import hashlib
import math
from typing import Any, List, Optional, Tuple
from uuid import uuid4
import numpy as np
import scipy.ndimage
from shapely.geometry import Point, Polygon

from src.domain.candidate import Candidate
from src.domain.enums import Capability
from src.domain.observation import ProcessedObservation
from src.domain.provenance import Provenance


class LocalCfarDetector:
    capability: Capability = Capability.IMPLEMENTED

    def __init__(
        self,
        guard_radius_px: Optional[int] = None,
        background_radius_px: Optional[int] = None,
        pfa: float = 1e-5,
        cfar_type: str = "CA",
        k_order_ratio: float = 0.75,
        min_training_cells: int = 8,
        min_cluster_size_px: int = 1,
        guard_radius_m: Optional[float] = None,
        background_radius_m: Optional[float] = None,
        gsd_m: float = 10.0,
    ) -> None:
        if gsd_m <= 0.0:
            raise ValueError("gsd_m must be strictly positive")

        if guard_radius_m is not None:
            g_px = max(1, int(round(guard_radius_m / gsd_m)))
        elif guard_radius_px is not None:
            g_px = guard_radius_px
        else:
            g_px = max(1, int(round(40.0 / gsd_m))) if background_radius_m is not None else 4

        if background_radius_m is not None:
            b_px = max(g_px + 1, int(round(background_radius_m / gsd_m)))
        elif background_radius_px is not None:
            b_px = background_radius_px
        else:
            b_px = max(g_px + 1, int(round(150.0 / gsd_m))) if guard_radius_m is not None else 15

        if g_px < 0:
            raise ValueError("guard_radius_px cannot be negative")
        if b_px <= g_px:
            raise ValueError("background_radius_px must be strictly greater than guard_radius_px")
        if not (0.0 < pfa < 1.0):
            raise ValueError("pfa must be in (0, 1)")

        self.gsd_m = float(gsd_m)
        self.guard_radius_px = int(g_px)
        self.background_radius_px = int(b_px)
        self.guard_radius_m = (
            float(guard_radius_m)
            if guard_radius_m is not None
            else float(self.guard_radius_px * self.gsd_m)
        )
        self.background_radius_m = (
            float(background_radius_m)
            if background_radius_m is not None
            else float(self.background_radius_px * self.gsd_m)
        )
        self.pfa = float(pfa)
        self.cfar_type = cfar_type.upper()
        self.k_order_ratio = float(k_order_ratio)
        self.min_training_cells = max(1, int(min_training_cells))
        self.min_cluster_size_px = max(1, int(min_cluster_size_px))

        offsets = []
        for dr in range(-self.background_radius_px, self.background_radius_px + 1):
            for dc in range(-self.background_radius_px, self.background_radius_px + 1):
                if not (abs(dr) <= self.guard_radius_px and abs(dc) <= self.guard_radius_px):
                    offsets.append((dr, dc))
        self.offsets = np.array(offsets, dtype=np.int32)

    def detect_mask(
        self, intensity: np.ndarray, land_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = intensity.shape
        valid = np.isfinite(intensity) & (intensity > 0.0)
        if land_mask is not None:
            valid = valid & (~land_mask.astype(bool))

        b = self.background_radius_px
        g = self.guard_radius_px

        masked_intensity = np.where(valid, intensity, 0.0).astype(np.float64)
        valid_float = valid.astype(np.float64)

        if self.cfar_type == "CA":
            bg_size = 2 * b + 1
            guard_size = 2 * g + 1

            box_bg_sum = scipy.ndimage.uniform_filter(
                masked_intensity, size=bg_size, mode="constant", cval=0.0
            ) * (bg_size**2)
            box_guard_sum = scipy.ndimage.uniform_filter(
                masked_intensity, size=guard_size, mode="constant", cval=0.0
            ) * (guard_size**2)
            annulus_sum = np.maximum(0.0, box_bg_sum - box_guard_sum)

            box_bg_cnt = scipy.ndimage.uniform_filter(
                valid_float, size=bg_size, mode="constant", cval=0.0
            ) * (bg_size**2)
            box_guard_cnt = scipy.ndimage.uniform_filter(
                valid_float, size=guard_size, mode="constant", cval=0.0
            ) * (guard_size**2)
            annulus_cnt = np.maximum(0.0, np.round(box_bg_cnt - box_guard_cnt)).astype(np.int64)

            valid_bg = annulus_cnt >= self.min_training_cells
            mu_bg = np.zeros_like(intensity, dtype=np.float64)
            mu_bg[valid_bg] = annulus_sum[valid_bg] / annulus_cnt[valid_bg]

            threshold = np.full_like(intensity, fill_value=np.inf, dtype=np.float64)
            n_vals = annulus_cnt[valid_bg]
            alpha = n_vals * (np.power(self.pfa, -1.0 / n_vals) - 1.0)
            threshold[valid_bg] = alpha * mu_bg[valid_bg]

            det_mask = valid & valid_bg & (intensity > threshold)
            snr_map = np.zeros_like(intensity, dtype=np.float64)
            snr_map[valid_bg] = intensity[valid_bg] / np.maximum(mu_bg[valid_bg], 1e-12)
            return det_mask, mu_bg, snr_map
        else:
            det_mask = np.zeros((h, w), dtype=bool)
            mu_bg = np.zeros((h, w), dtype=np.float64)
            snr_map = np.zeros((h, w), dtype=np.float64)

            padded_intensity = np.pad(masked_intensity, b, mode="constant", constant_values=0.0)
            padded_valid = np.pad(valid, b, mode="constant", constant_values=False)

            r_indices, c_indices = np.where(valid)
            for r, c in zip(r_indices, c_indices):
                pr = r + b
                pc = c + b
                sample_r = pr + self.offsets[:, 0]
                sample_c = pc + self.offsets[:, 1]
                cell_valid = padded_valid[sample_r, sample_c]
                valid_count = int(np.sum(cell_valid))
                if valid_count < self.min_training_cells:
                    continue

                train_cells = padded_intensity[sample_r[cell_valid], sample_c[cell_valid]]
                n = len(train_cells)
                k_idx = int(round(self.k_order_ratio * (n - 1)))
                rep_val = float(np.partition(train_cells, k_idx)[k_idx])
                mu_bg[r, c] = rep_val

                alpha = n * (np.power(self.pfa, -1.0 / n) - 1.0)
                thresh = rep_val * alpha
                if intensity[r, c] > thresh:
                    det_mask[r, c] = True
                snr_map[r, c] = intensity[r, c] / max(rep_val, 1e-12)

            return det_mask, mu_bg, snr_map

    def detect(
        self,
        intensity: np.ndarray,
        processed_observation: Optional[ProcessedObservation] = None,
        land_mask: Optional[np.ndarray] = None,
        pixel_spacing_m: float = 10.0,
        affine_transform: Optional[Any] = None,
    ) -> Tuple[Candidate, ...]:
        det_mask, mu_bg, _ = self.detect_mask(intensity, land_mask=land_mask)
        labeled, num_features = scipy.ndimage.label(det_mask)
        if num_features == 0:
            return ()

        candidates: List[Candidate] = []
        for lbl in range(1, num_features + 1):
            coords = np.argwhere(labeled == lbl)
            if len(coords) < self.min_cluster_size_px:
                continue

            min_row = int(np.min(coords[:, 0]))
            max_row = int(np.max(coords[:, 0]))
            min_col = int(np.min(coords[:, 1]))
            max_col = int(np.max(coords[:, 1]))
            geom_px = (float(min_col), float(min_row), float(max_col), float(max_row))

            vals = intensity[coords[:, 0], coords[:, 1]]
            peak_val = float(np.max(vals))
            mean_val = float(np.mean(vals))
            c_row = float(np.mean(coords[:, 0]))
            c_col = float(np.mean(coords[:, 1]))

            bg_val = float(mu_bg[int(round(c_row)), int(round(c_col))])
            snr = peak_val / max(bg_val, 1e-12)

            if len(coords) >= 3:
                centered = coords - np.array([c_row, c_col])
                cov = np.cov(centered, rowvar=False)
                if cov.shape == (2, 2) and np.all(np.isfinite(cov)):
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    sort_idx = np.argsort(eigvals)[::-1]
                    eigvals = eigvals[sort_idx]
                    eigvecs = eigvecs[:, sort_idx]
                    primary_vec = eigvecs[:, 0]
                    angle_rad = math.atan2(primary_vec[1], -primary_vec[0])
                    orientation_deg = (math.degrees(angle_rad) + 360.0) % 360.0
                    length_px = float(2.0 * math.sqrt(max(0.25, eigvals[0])) * 2.0)
                    width_px = float(2.0 * math.sqrt(max(0.25, eigvals[1])) * 2.0)
                else:
                    orientation_deg = 0.0
                    length_px = float(max_row - min_row + 1)
                    width_px = float(max_col - min_col + 1)
            else:
                orientation_deg = 0.0
                length_px = float(len(coords))
                width_px = 1.0

            length_m = max(1.0, length_px * pixel_spacing_m)
            width_m = max(1.0, width_px * pixel_spacing_m)
            if width_m > length_m:
                length_m, width_m = width_m, length_m
            aspect_ratio = length_m / width_m

            if affine_transform is not None and hasattr(affine_transform, "pixel_to_geo"):
                geo_x, geo_y = affine_transform.pixel_to_geo(c_col, c_row)
                geom_wgs84 = Point(geo_x, geo_y)
            else:
                geom_wgs84 = Point(c_col, c_row)

            features = (
                ("snr", float(snr)),
                ("peak_intensity", float(peak_val)),
                ("mean_intensity", float(mean_val)),
                ("pixel_count", float(len(coords))),
                ("centroid_row", float(c_row)),
                ("centroid_col", float(c_col)),
                ("length_m", float(length_m)),
                ("width_m", float(width_m)),
                ("orientation_deg", float(orientation_deg)),
                ("aspect_ratio", float(aspect_ratio)),
            )

            parent_prov_id = (
                processed_observation.provenance.provenance_id
                if processed_observation
                else None
            )
            src_asset = (
                processed_observation.provenance.source_asset_id
                if processed_observation
                else "synthetic_sar"
            )
            meta_ref = (
                processed_observation.provenance.acquisition_metadata_ref
                if processed_observation
                else ""
            )

            cand_prov = Provenance(
                provenance_id=f"prov-cand-{uuid4().hex[:12]}",
                source_asset_id=src_asset,
                acquisition_metadata_ref=meta_ref,
                pipeline_version="3.0.0",
                git_commit=(
                    processed_observation.provenance.git_commit
                    if processed_observation
                    else "abcdef0"
                ),
                config_hash=(
                    processed_observation.provenance.config_hash
                    if processed_observation
                    else "hash0"
                ),
                model_id=None,
                model_version=None,
                input_checksum=(
                    processed_observation.provenance.output_checksum
                    if processed_observation
                    else "input_raw"
                ),
                output_checksum=hashlib.sha256(f"cand_{lbl}".encode("utf-8")).hexdigest()[:16],
                processing_run_id=f"run-{uuid4().hex[:8]}",
                processing_time_utc=datetime.now(timezone.utc),
                parent_provenance_id=parent_prov_id,
            )

            cand = Candidate(
                candidate_id=f"cand-{uuid4().hex[:12]}",
                processed_id=(
                    processed_observation.processed_id
                    if processed_observation
                    else "proc-synthetic"
                ),
                geom_px=geom_px,
                geom_wgs84=geom_wgs84,
                generator="LocalCfarDetector",
                generator_capability=Capability.IMPLEMENTED,
                features=features,
                quality_flags=(),
                provenance=cand_prov,
            )
            candidates.append(cand)

        return tuple(candidates)
