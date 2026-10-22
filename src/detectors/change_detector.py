import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import yaml
from scipy.ndimage import label, find_objects
from shapely.geometry import Polygon

from src.core.contracts import Detection
from src.core.coords import AffineCoordinateTransformer
from src.detectors.base import BaseDetector

try:
    import rasterio
except ImportError:
    rasterio = None


class DualStreamChangeDetector(BaseDetector):
    def __init__(
        self,
        weights_config_path: Optional[str] = "configs/model_registry.yaml",
        model_name: str = "ds_unet",
        pixel_spacing_m: float = 10.0,
        default_crs: str = "EPSG:4326",
        diff_threshold: float = 0.35,
        cosine_threshold: float = 0.98,
        min_area_px: int = 10,
        structure_area_threshold_m2: float = 2500.0,
    ) -> None:
        self.model_name = model_name
        self.pixel_spacing_m = pixel_spacing_m
        self.default_crs = default_crs
        self.diff_threshold = diff_threshold
        self.cosine_threshold = cosine_threshold
        self.min_area_px = min_area_px
        self.structure_area_threshold_m2 = structure_area_threshold_m2
        self.model = None
        self.mode = "heuristic_diff"

        if weights_config_path and Path(weights_config_path).is_file():
            with open(weights_config_path, "r", encoding="utf-8") as f:
                registry = yaml.safe_load(f)
            model_info = registry.get("models", {}).get(model_name)
            if model_info:
                local_path = model_info.get("local_path")
                if local_path and Path(local_path).is_file():
                    self.mode = "model_loaded"

    def _to_2d_array(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4):
                arr = np.mean(arr, axis=0)
            else:
                arr = np.mean(arr, axis=2)
        return arr.astype(np.float32)

    def _compute_cosine_suppression(
        self, t1: np.ndarray, t2: np.ndarray, window_size: int = 5
    ) -> np.ndarray:
        pad = window_size // 2
        p_t1 = np.pad(t1, pad, mode="reflect")
        p_t2 = np.pad(t2, pad, mode="reflect")

        shape = (t1.shape[0], t1.shape[1], window_size, window_size)
        strides = (
            p_t1.strides[0],
            p_t1.strides[1],
            p_t1.strides[0],
            p_t1.strides[1],
        )

        w1 = np.lib.stride_tricks.as_strided(
            p_t1, shape=shape, strides=strides, writeable=False
        ).reshape(t1.shape[0], t1.shape[1], -1)
        w2 = np.lib.stride_tricks.as_strided(
            p_t2, shape=shape, strides=strides, writeable=False
        ).reshape(t2.shape[0], t2.shape[1], -1)

        dot = np.sum(w1 * w2, axis=-1)
        norm1 = np.linalg.norm(w1, axis=-1)
        norm2 = np.linalg.norm(w2, axis=-1)

        cos_map = np.ones_like(t1, dtype=np.float32)
        valid = (norm1 > 1e-6) & (norm2 > 1e-6)
        cos_map[valid] = dot[valid] / (norm1[valid] * norm2[valid])

        return cos_map

    def detect(
        self,
        image_input: Union[Tuple[np.ndarray, np.ndarray], Tuple[str, str]],
        sensor: str = "S1_RTC",
        source_tile: str = "bitemporal_scene",
        acquisition_time: Optional[datetime] = None,
        transform_matrix: Optional[Tuple[float, float, float, float, float, float]] = None,
        confidence_heuristic: float = 0.85,
        apply_cosine_filter: bool = False,
        **kwargs: Any,
    ) -> List[Detection]:
        if acquisition_time is None:
            acquisition_time = datetime.now(timezone.utc)

        crs_str = self.default_crs

        if isinstance(image_input[0], str):
            if rasterio is None:
                raise ImportError("rasterio is required to read raster files from path.")
            with rasterio.open(image_input[0]) as s1:
                t1_raw = s1.read()
                affine_transform = s1.transform
                transform_matrix = (
                    affine_transform.a,
                    affine_transform.b,
                    affine_transform.c,
                    affine_transform.d,
                    affine_transform.e,
                    affine_transform.f,
                )
                if s1.crs:
                    crs_str = str(s1.crs)
            with rasterio.open(image_input[1]) as s2:
                t2_raw = s2.read()
        else:
            t1_raw, t2_raw = image_input

        t1 = self._to_2d_array(t1_raw)
        t2 = self._to_2d_array(t2_raw)

        if t1.shape != t2.shape:
            raise ValueError(f"Shape mismatch between bi-temporal images: {t1.shape} vs {t2.shape}")

        if transform_matrix is None:
            transform_matrix = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        coord_transformer = AffineCoordinateTransformer(transform_matrix)

        abs_diff = np.abs(t2 - t1)
        change_mask = abs_diff >= self.diff_threshold

        if apply_cosine_filter:
            cos_map = self._compute_cosine_suppression(t1, t2)
            change_mask = change_mask & (cos_map < self.cosine_threshold)

        labeled_mask, num_features = label(change_mask)
        slices = find_objects(labeled_mask)

        results: List[Detection] = []
        det_idx = 1
        pixel_area_m2 = self.pixel_spacing_m * self.pixel_spacing_m

        for label_val, slc in enumerate(slices, start=1):
            if slc is None:
                continue

            r_slice, c_slice = slc
            blob_patch = (labeled_mask[r_slice, c_slice] == label_val)
            area_px = int(np.sum(blob_patch))

            if area_px < self.min_area_px:
                continue

            r_min, r_max = r_slice.start, r_slice.stop
            c_min, c_max = c_slice.start, c_slice.stop
            geom_px = (float(c_min), float(r_min), float(c_max), float(r_max))

            p1 = coord_transformer.pixel_to_geo(c_min, r_min)
            p2 = coord_transformer.pixel_to_geo(c_max, r_min)
            p3 = coord_transformer.pixel_to_geo(c_max, r_max)
            p4 = coord_transformer.pixel_to_geo(c_min, r_max)
            geom_wgs84 = Polygon([p1, p2, p3, p4, p1])

            area_m2 = area_px * pixel_area_m2
            target_class = (
                "new_structure"
                if area_m2 >= self.structure_area_threshold_m2
                else "unknown_structure"
            )
            estimated_length_m = math.sqrt(area_m2)

            det = Detection(
                det_id=f"CHANGE_DET_{source_tile}_{det_idx:04d}",
                sensor=sensor,
                geom_px=geom_px,
                geom_wgs84=geom_wgs84,
                crs=crs_str,
                confidence=confidence_heuristic,
                length_m=estimated_length_m,
                heading_deg=None,
                target_class=target_class,
                source_tile=source_tile,
                acquisition_time=acquisition_time,
            )
            results.append(det)
            det_idx += 1

        return results
