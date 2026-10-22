import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import yaml
from scipy.ndimage import label, find_objects
from shapely.geometry import Polygon

from src.core.cfar import OSCFARDetector
from src.core.contracts import Detection
from src.core.coords import AffineCoordinateTransformer
from src.detectors.base import BaseDetector

try:
    import rasterio
except ImportError:
    rasterio = None


class SarVesselDetector(BaseDetector):
    def __init__(
        self,
        weights_config_path: Optional[str] = "configs/model_registry.yaml",
        cfar_detector: Optional[OSCFARDetector] = None,
        model_name: str = "sar_vessel_detect",
        pixel_spacing_m: float = 10.0,
        default_crs: str = "EPSG:4326",
    ) -> None:
        self.model_name = model_name
        self.pixel_spacing_m = pixel_spacing_m
        self.default_crs = default_crs
        self.cfar = cfar_detector or OSCFARDetector(
            guard_m=50.0,
            train_m=200.0,
            pixel_spacing_m=pixel_spacing_m,
            pfa=1e-5,
            k_order_ratio=0.75,
        )
        self.model = None
        self.mode = "heuristic_cfar"

        if weights_config_path and Path(weights_config_path).is_file():
            with open(weights_config_path, "r", encoding="utf-8") as f:
                registry = yaml.safe_load(f)
            model_info = registry.get("models", {}).get(model_name)
            if model_info:
                local_path = model_info.get("local_path")
                if local_path and Path(local_path).is_file():
                    self.mode = "model_loaded"

    def _calculate_heading_and_length(
        self, blob_mask: np.ndarray
    ) -> Tuple[Optional[float], Optional[float]]:
        coords = np.argwhere(blob_mask)
        if len(coords) < 3:
            return None, None

        mean = np.mean(coords, axis=0)
        centered = coords - mean
        cov = np.cov(centered, rowvar=False)

        if cov.shape != (2, 2) or np.any(np.isnan(cov)):
            return None, None

        eigvals, eigvecs = np.linalg.eigh(cov)
        sort_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sort_indices]
        eigvecs = eigvecs[:, sort_indices]

        primary_vec = eigvecs[:, 0]
        row_dir, col_dir = primary_vec[0], primary_vec[1]
        angle_rad = math.atan2(col_dir, -row_dir)
        heading_deg = (math.degrees(angle_rad) + 360.0) % 360.0

        length_px = float(2.0 * math.sqrt(max(0.0, eigvals[0])) * 2.0)
        length_m = length_px * self.pixel_spacing_m

        return heading_deg, length_m

    def detect(
        self,
        image_input: Union[str, np.ndarray],
        sensor: str = "S1_GRD",
        source_tile: str = "unknown_scene",
        acquisition_time: Optional[datetime] = None,
        transform_matrix: Optional[Tuple[float, float, float, float, float, float]] = None,
        land_mask: Optional[np.ndarray] = None,
        confidence_heuristic: float = 0.95,
        **kwargs: Any,
    ) -> List[Detection]:
        if acquisition_time is None:
            acquisition_time = datetime.now(timezone.utc)

        crs_str = self.default_crs
        if isinstance(image_input, str):
            if rasterio is None:
                raise ImportError("rasterio is required to read raster files from path.")
            with rasterio.open(image_input) as src:
                intensity = src.read(1).astype(np.float32)
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
            intensity = image_input.read(1).astype(np.float32)
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
            intensity = image_input.astype(np.float32)

        if transform_matrix is None:
            transform_matrix = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        coord_transformer = AffineCoordinateTransformer(transform_matrix)
        detections_mask = self.cfar.detect(intensity, land_mask=land_mask)

        labeled_mask, num_features = label(detections_mask)
        slices = find_objects(labeled_mask)

        results: List[Detection] = []

        for idx, slc in enumerate(slices, start=1):
            if slc is None:
                continue

            row_slice, col_slice = slc
            blob_patch = (labeled_mask[row_slice, col_slice] == idx)

            r_min, r_max = row_slice.start, row_slice.stop
            c_min, c_max = col_slice.start, col_slice.stop

            geom_px = (float(c_min), float(r_min), float(c_max), float(r_max))

            heading_deg, length_m = self._calculate_heading_and_length(blob_patch)

            geo_p1 = coord_transformer.pixel_to_geo(c_min, r_min)
            geo_p2 = coord_transformer.pixel_to_geo(c_max, r_min)
            geo_p3 = coord_transformer.pixel_to_geo(c_max, r_max)
            geo_p4 = coord_transformer.pixel_to_geo(c_min, r_max)

            geom_wgs84 = Polygon([geo_p1, geo_p2, geo_p3, geo_p4, geo_p1])

            det_id = f"SAR_DET_{source_tile}_{idx:04d}"

            results.append(
                Detection(
                    det_id=det_id,
                    sensor=sensor,
                    geom_px=geom_px,
                    geom_wgs84=geom_wgs84,
                    crs=crs_str,
                    confidence=confidence_heuristic,
                    length_m=length_m,
                    heading_deg=heading_deg,
                    target_class="vessel",
                    source_tile=source_tile,
                    acquisition_time=acquisition_time,
                )
            )

        return results
