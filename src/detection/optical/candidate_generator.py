from datetime import datetime, timezone
import hashlib
import math
from typing import Any, List, Optional, Tuple
from uuid import uuid4
import numpy as np
import scipy.ndimage
from shapely.geometry import Point

from src.domain.candidate import Candidate
from src.domain.enums import Capability
from src.domain.geometry import AxisAlignedBoundingBox
from src.domain.observation import ProcessedObservation
from src.domain.provenance import Provenance


class HeuristicBrightBlobCandidateGenerator:
    capability: Capability = Capability.HEURISTIC

    def __init__(
        self,
        threshold: float = 0.5,
        min_area_px: int = 4,
        max_area_px: int = 5000,
        pixel_spacing_m: float = 0.5,
    ) -> None:
        self.threshold = threshold
        self.min_area_px = min_area_px
        self.max_area_px = max_area_px
        self.pixel_spacing_m = pixel_spacing_m

    def _prepare_intensity(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.astype(np.float32)
        if image.ndim == 3:
            if image.shape[0] in (1, 3, 4) and image.shape[0] < image.shape[1]:
                return np.mean(image.astype(np.float32), axis=0)
            return np.mean(image.astype(np.float32), axis=-1)
        raise ValueError(f"Unsupported image array dimensions: {image.ndim}")

    def generate(
        self,
        image: np.ndarray,
        processed_observation: Optional[ProcessedObservation] = None,
        pixel_spacing_m: Optional[float] = None,
        affine_transform: Optional[Any] = None,
    ) -> Tuple[Candidate, ...]:
        spacing = (
            pixel_spacing_m
            if pixel_spacing_m is not None
            else self.pixel_spacing_m
        )
        intensity = self._prepare_intensity(image)

        binary_mask = intensity >= self.threshold
        labeled, num_features = scipy.ndimage.label(binary_mask)
        if num_features == 0:
            return ()

        candidates: List[Candidate] = []
        for lbl in range(1, num_features + 1):
            coords = np.argwhere(labeled == lbl)
            num_px = len(coords)
            if num_px < self.min_area_px or num_px > self.max_area_px:
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
            geom_px = aabb.to_tuple()

            c_row = float(np.mean(coords[:, 0]))
            c_col = float(np.mean(coords[:, 1]))

            vals = intensity[coords[:, 0], coords[:, 1]]
            mean_val = float(np.mean(vals))
            max_val = float(np.max(vals))

            if num_px >= 3:
                centered = coords - np.array([c_row, c_col])
                cov = np.cov(centered, rowvar=False)
                if cov.shape == (2, 2) and np.all(np.isfinite(cov)):
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    sort_idx = np.argsort(eigvals)[::-1]
                    eigvals = eigvals[sort_idx]
                    eigvecs = eigvecs[:, sort_idx]
                    primary_vec = eigvecs[:, 0]
                    angle_rad = math.atan2(primary_vec[1], -primary_vec[0])
                    orientation_deg = (
                        math.degrees(angle_rad) + 360.0
                    ) % 360.0
                    length_px = float(
                        2.0 * math.sqrt(max(0.25, eigvals[0])) * 2.0
                    )
                    width_px = float(
                        2.0 * math.sqrt(max(0.25, eigvals[1])) * 2.0
                    )
                else:
                    orientation_deg = 0.0
                    length_px = float(max_row - min_row + 1)
                    width_px = float(max_col - min_col + 1)
            else:
                orientation_deg = 0.0
                length_px = float(num_px)
                width_px = 1.0

            length_m = max(1.0, length_px * spacing)
            width_m = max(1.0, width_px * spacing)
            if width_m > length_m:
                length_m, width_m = width_m, length_m
            aspect_ratio = length_m / width_m

            if affine_transform is not None and hasattr(
                affine_transform, "pixel_to_geo"
            ):
                geo_x, geo_y = affine_transform.pixel_to_geo(c_col, c_row)
                geom_wgs84 = Point(geo_x, geo_y)
            else:
                geom_wgs84 = Point(c_col, c_row)

            features = (
                ("brightness_mean", float(mean_val)),
                ("brightness_max", float(max_val)),
                ("pixel_count", float(num_px)),
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
                else "optical_array"
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
                    else "unknown"
                ),
                config_hash=(
                    processed_observation.provenance.config_hash
                    if processed_observation
                    else "unknown"
                ),
                model_id=None,
                model_version=None,
                input_checksum=(
                    processed_observation.provenance.output_checksum
                    if processed_observation
                    else hashlib.sha256(
                        intensity.tobytes()[:64]
                    ).hexdigest()[:16]
                ),
                output_checksum=hashlib.sha256(
                    f"{c_row}_{c_col}_{num_px}".encode("utf-8")
                ).hexdigest()[:16],
                processing_run_id=(
                    processed_observation.provenance.processing_run_id
                    if processed_observation
                    else f"run-{uuid4().hex[:8]}"
                ),
                processing_time_utc=datetime.now(timezone.utc),
                parent_provenance_id=parent_prov_id,
            )

            candidates.append(
                Candidate(
                    candidate_id=f"cand-blob-{uuid4().hex[:12]}",
                    processed_id=(
                        processed_observation.processed_id
                        if processed_observation
                        else "proc-unspecified"
                    ),
                    geom_px=geom_px,
                    geom_wgs84=geom_wgs84,
                    generator="HeuristicBrightBlobCandidateGenerator",
                    generator_capability=Capability.HEURISTIC,
                    features=features,
                    quality_flags=(),
                    provenance=cand_prov,
                    bounding_box=aabb,
                )
            )

        return tuple(candidates)
