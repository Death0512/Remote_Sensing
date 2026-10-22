import math
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from scipy.ndimage import label, center_of_mass
from shapely.geometry import Point, Polygon

from src.core.contracts import Detection
from src.core.coords import AffineCoordinateTransformer
from src.detectors.base import BaseDetector

try:
    import rasterio
except ImportError:
    rasterio = None


class ParallaxInflightDetector(BaseDetector):
    def __init__(
        self,
        pixel_spacing_m: float = 10.0,
        inter_band_delay_seconds: float = 0.52,
        min_velocity_kmh: float = 300.0,
        max_velocity_kmh: float = 1000.0,
        triplet_colinearity_max_deg: float = 25.0,
        min_displacement_px: float = 3.0,
        max_displacement_px: float = 20.0,
        default_crs: str = "EPSG:4326",
    ) -> None:
        self.pixel_spacing_m = pixel_spacing_m
        self.inter_band_delay_seconds = inter_band_delay_seconds
        self.min_velocity_kmh = min_velocity_kmh
        self.max_velocity_kmh = max_velocity_kmh
        self.triplet_colinearity_max_deg = triplet_colinearity_max_deg
        self.min_displacement_px = min_displacement_px
        self.max_displacement_px = max_displacement_px
        self.default_crs = default_crs

    def _extract_peaks(
        self, band: np.ndarray, threshold: float
    ) -> List[Tuple[float, float, float]]:
        mask = band >= threshold
        labeled, num_features = label(mask)
        if num_features == 0:
            return []

        centers = center_of_mass(mask, labeled, range(1, num_features + 1))
        peaks: List[Tuple[float, float, float]] = []
        for idx, (r, c) in enumerate(centers, start=1):
            val = float(band[int(round(r)), int(round(c))])
            peaks.append((float(c), float(r), val))
        return peaks

    def _angle_between_vectors(
        self, v1: Tuple[float, float], v2: Tuple[float, float]
    ) -> float:
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        norm1 = math.hypot(v1[0], v1[1])
        norm2 = math.hypot(v2[0], v2[1])
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 180.0
        cos_theta = max(-1.0, min(1.0, dot / (norm1 * norm2)))
        return math.degrees(math.acos(cos_theta))

    def detect(
        self,
        image_input: Union[str, np.ndarray],
        sensor: str = "S2_L1C",
        source_tile: str = "s2_scene",
        acquisition_time: Optional[datetime] = None,
        transform_matrix: Optional[Tuple[float, float, float, float, float, float]] = None,
        confidence_heuristic: float = 0.88,
        peak_threshold: float = 0.8,
        **kwargs: Any,
    ) -> List[Detection]:
        if acquisition_time is None:
            acquisition_time = datetime.now(timezone.utc)

        crs_str = self.default_crs

        if isinstance(image_input, str):
            if rasterio is None:
                raise ImportError("rasterio is required to read raster files from path.")
            with rasterio.open(image_input) as src:
                arr = src.read()
                affine_transform = src.transform
                transform_matrix = (
                    affine_transform.a,
                    affine_transform.b,
                    affine_transform.c,
                    affine_transform.d,
                    affine_transform.e,
                    affine_transform.f,
                )
                if src.crs:
                    crs_str = str(src.crs)
        elif hasattr(image_input, "read"):
            arr = image_input.read()
            affine_transform = image_input.transform
            transform_matrix = (
                affine_transform.a,
                affine_transform.b,
                affine_transform.c,
                affine_transform.d,
                affine_transform.e,
                affine_transform.f,
            )
            if image_input.crs:
                crs_str = str(image_input.crs)
        else:
            arr = image_input

        if arr.ndim != 3 or arr.shape[0] < 3:
            raise ValueError("Parallax detector requires multi-band array with at least 3 channels (RGB).")

        red_band = arr[0].astype(np.float32)
        green_band = arr[1].astype(np.float32)
        blue_band = arr[2].astype(np.float32)

        if transform_matrix is None:
            transform_matrix = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        coord_transformer = AffineCoordinateTransformer(transform_matrix)

        r_peaks = self._extract_peaks(red_band, peak_threshold)
        g_peaks = self._extract_peaks(green_band, peak_threshold)
        b_peaks = self._extract_peaks(blue_band, peak_threshold)

        results: List[Detection] = []
        det_idx = 1

        for rx, ry, _ in r_peaks:
            for gx, gy, _ in g_peaks:
                rg_dx = gx - rx
                rg_dy = gy - ry
                rg_dist = math.hypot(rg_dx, rg_dy)

                if not (self.min_displacement_px <= rg_dist <= self.max_displacement_px):
                    continue

                for bx, by, _ in b_peaks:
                    gb_dx = bx - gx
                    gb_dy = by - gy
                    gb_dist = math.hypot(gb_dx, gb_dy)

                    if not (self.min_displacement_px <= gb_dist <= self.max_displacement_px):
                        continue

                    angle_diff = self._angle_between_vectors((rg_dx, rg_dy), (gb_dx, gb_dy))
                    if angle_diff > self.triplet_colinearity_max_deg:
                        continue

                    dist_ratio = rg_dist / max(1e-5, gb_dist)
                    if not (0.5 <= dist_ratio <= 2.0):
                        continue

                    mean_disp_px = (rg_dist + gb_dist) / 2.0
                    velocity_mps = (mean_disp_px * self.pixel_spacing_m) / self.inter_band_delay_seconds
                    velocity_kmh = velocity_mps * 3.6

                    if not (self.min_velocity_kmh <= velocity_kmh <= self.max_velocity_kmh):
                        continue

                    overall_dx = bx - rx
                    overall_dy = by - ry
                    heading_rad = math.atan2(overall_dx, -overall_dy)
                    heading_deg = (math.degrees(heading_rad) + 360.0) % 360.0

                    c_min = min(rx, gx, bx) - 2.0
                    c_max = max(rx, gx, bx) + 2.0
                    r_min = min(ry, gy, by) - 2.0
                    r_max = max(ry, gy, by) + 2.0

                    geom_px = (float(c_min), float(r_min), float(c_max), float(r_max))

                    p1 = coord_transformer.pixel_to_geo(c_min, r_min)
                    p2 = coord_transformer.pixel_to_geo(c_max, r_min)
                    p3 = coord_transformer.pixel_to_geo(c_max, r_max)
                    p4 = coord_transformer.pixel_to_geo(c_min, r_max)
                    geom_wgs84 = Polygon([p1, p2, p3, p4, p1])

                    length_m = mean_disp_px * self.pixel_spacing_m

                    det = Detection(
                        det_id=f"PARALLAX_DET_{source_tile}_{det_idx:04d}",
                        sensor=sensor,
                        geom_px=geom_px,
                        geom_wgs84=geom_wgs84,
                        crs=crs_str,
                        confidence=confidence_heuristic,
                        length_m=length_m,
                        heading_deg=heading_deg,
                        target_class="aircraft_inflight",
                        source_tile=source_tile,
                        acquisition_time=acquisition_time,
                    )
                    results.append(det)
                    det_idx += 1

        return results
