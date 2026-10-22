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

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class OpticalAircraftDetector(BaseDetector):
    def __init__(
        self,
        weights_config_path: Optional[str] = "configs/model_registry.yaml",
        model_name: str = "yolov8_obb",
        pixel_spacing_m: float = 0.5,
        default_crs: str = "EPSG:4326",
        intensity_threshold: float = 0.6,
        min_area_px: int = 15,
        max_area_px: int = 5000,
    ) -> None:
        self.model_name = model_name
        self.pixel_spacing_m = pixel_spacing_m
        self.default_crs = default_crs
        self.intensity_threshold = intensity_threshold
        self.min_area_px = min_area_px
        self.max_area_px = max_area_px
        self.model = None
        self.mode = "heuristic_blob"

        if weights_config_path and Path(weights_config_path).is_file():
            with open(weights_config_path, "r", encoding="utf-8") as f:
                registry = yaml.safe_load(f)
            model_info = registry.get("models", {}).get(model_name)
            if model_info:
                local_path = model_info.get("local_path")
                if local_path and Path(local_path).is_file() and YOLO is not None:
                    try:
                        self.model = YOLO(local_path)
                        self.mode = "model_loaded"
                    except Exception:
                        self.mode = "heuristic_blob"

    def _compute_obb_from_mask(
        self, mask: np.ndarray, offset_row: int, offset_col: int
    ) -> Optional[Tuple[float, float, float, float, float]]:
        coords = np.argwhere(mask)
        if len(coords) < 4:
            return None

        coords_global = coords + np.array([offset_row, offset_col])
        center_row = float(np.mean(coords_global[:, 0]))
        center_col = float(np.mean(coords_global[:, 1]))

        centered = coords_global - np.array([center_row, center_col])
        cov = np.cov(centered, rowvar=False)
        if cov.shape != (2, 2) or np.any(np.isnan(cov)):
            return None

        eigvals, eigvecs = np.linalg.eigh(cov)
        sort_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sort_indices]
        eigvecs = eigvecs[:, sort_indices]

        major_axis = eigvecs[:, 0]
        angle_rad = math.atan2(major_axis[0], major_axis[1])
        angle_deg = (math.degrees(angle_rad) + 360.0) % 360.0

        length_px = float(2.0 * math.sqrt(max(0.1, eigvals[0])) * 2.0)
        width_px = float(2.0 * math.sqrt(max(0.1, eigvals[1])) * 2.0)

        return center_col, center_row, width_px, length_px, angle_deg

    def _obb_to_polygon(
        self,
        center_col: float,
        center_row: float,
        width: float,
        height: float,
        angle_deg: float,
        transformer: AffineCoordinateTransformer,
    ) -> Polygon:
        theta = math.radians(angle_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        hw = width / 2.0
        hh = height / 2.0

        local_corners = [
            (-hw, -hh),
            (hw, -hh),
            (hw, hh),
            (-hw, hh),
        ]

        geo_corners = []
        for lx, ly in local_corners:
            px = center_col + (lx * cos_t - ly * sin_t)
            py = center_row + (lx * sin_t + ly * cos_t)
            gx, gy = transformer.pixel_to_geo(px, py)
            geo_corners.append((gx, gy))

        geo_corners.append(geo_corners[0])
        return Polygon(geo_corners)

    def _detect_with_model(
        self,
        arr: np.ndarray,
        sensor: str,
        source_tile: str,
        acquisition_time: datetime,
        coord_transformer: AffineCoordinateTransformer,
        crs_str: str,
    ) -> List[Detection]:
        if arr.ndim == 2:
            img = np.repeat(arr[:, :, None], 3, axis=2)
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4):
                img = np.transpose(arr[:3], (1, 2, 0))
            else:
                img = arr[:, :, :3]
        else:
            return []

        if img.dtype != np.uint8:
            if np.max(img) <= 1.0:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        results = self.model.predict(img, verbose=False)
        detections: List[Detection] = []
        det_idx = 1

        for res in results:
            obb = getattr(res, "obb", None)
            if obb is None or len(obb) == 0:
                continue

            xywhr = obb.xywhr
            confs = obb.conf
            if hasattr(xywhr, "cpu"):
                xywhr = xywhr.cpu().numpy()
            if hasattr(confs, "cpu"):
                confs = confs.cpu().numpy()

            for i, box in enumerate(xywhr):
                c_col, c_row, w_px, h_px, r_rad = box
                angle_deg = (math.degrees(float(r_rad)) + 360.0) % 360.0
                poly = self._obb_to_polygon(
                    float(c_col), float(c_row), float(w_px), float(h_px), angle_deg, coord_transformer
                )
                r_min = float(c_row - h_px / 2.0)
                r_max = float(c_row + h_px / 2.0)
                c_min = float(c_col - w_px / 2.0)
                c_max = float(c_col + w_px / 2.0)
                geom_px = (c_min, r_min, c_max, r_max)
                length_m = max(float(w_px), float(h_px)) * self.pixel_spacing_m
                conf_val = float(confs[i]) if i < len(confs) else 0.85

                det = Detection(
                    det_id=f"OPT_DET_{source_tile}_{det_idx:04d}",
                    sensor=sensor,
                    geom_px=geom_px,
                    geom_wgs84=poly,
                    crs=crs_str,
                    confidence=conf_val,
                    length_m=length_m,
                    heading_deg=angle_deg,
                    target_class="aircraft_parked",
                    source_tile=source_tile,
                    acquisition_time=acquisition_time,
                )
                detections.append(det)
                det_idx += 1

        return detections

    def _detect_heuristic(
        self,
        arr: np.ndarray,
        sensor: str,
        source_tile: str,
        acquisition_time: datetime,
        coord_transformer: AffineCoordinateTransformer,
        crs_str: str,
        confidence_heuristic: float,
    ) -> List[Detection]:
        if arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4):
                intensity = np.mean(arr[:3], axis=0).astype(np.float32)
            else:
                intensity = np.mean(arr[:, :, :3], axis=2).astype(np.float32)
        elif arr.ndim == 2:
            intensity = arr.astype(np.float32)
        else:
            raise ValueError(f"Unsupported array shape {arr.shape}")

        detections_mask = intensity >= self.intensity_threshold
        labeled_mask, _ = label(detections_mask)
        slices = find_objects(labeled_mask)

        results: List[Detection] = []
        det_idx = 1

        for label_val, slc in enumerate(slices, start=1):
            if slc is None:
                continue

            r_slice, c_slice = slc
            blob_patch = (labeled_mask[r_slice, c_slice] == label_val)
            area = int(np.sum(blob_patch))

            if not (self.min_area_px <= area <= self.max_area_px):
                continue

            obb = self._compute_obb_from_mask(blob_patch, r_slice.start, c_slice.start)
            if obb is None:
                continue

            c_col, c_row, w_px, h_px, angle = obb
            poly = self._obb_to_polygon(c_col, c_row, w_px, h_px, angle, coord_transformer)

            r_min, r_max = r_slice.start, r_slice.stop
            c_min, c_max = c_slice.start, c_slice.stop
            geom_px = (float(c_min), float(r_min), float(c_max), float(r_max))

            length_m = max(w_px, h_px) * self.pixel_spacing_m

            det = Detection(
                det_id=f"OPT_DET_{source_tile}_{det_idx:04d}",
                sensor=sensor,
                geom_px=geom_px,
                geom_wgs84=poly,
                crs=crs_str,
                confidence=confidence_heuristic,
                length_m=length_m,
                heading_deg=angle,
                target_class="aircraft_parked",
                source_tile=source_tile,
                acquisition_time=acquisition_time,
            )
            results.append(det)
            det_idx += 1

        return results

    def detect(
        self,
        image_input: Union[str, np.ndarray],
        sensor: str = "VHR",
        source_tile: str = "optical_scene",
        acquisition_time: Optional[datetime] = None,
        transform_matrix: Optional[Tuple[float, float, float, float, float, float]] = None,
        confidence_heuristic: float = 0.90,
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

        if transform_matrix is None:
            transform_matrix = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        coord_transformer = AffineCoordinateTransformer(transform_matrix)

        if self.model is not None and self.mode == "model_loaded":
            try:
                model_dets = self._detect_with_model(
                    arr=arr,
                    sensor=sensor,
                    source_tile=source_tile,
                    acquisition_time=acquisition_time,
                    coord_transformer=coord_transformer,
                    crs_str=crs_str,
                )
                if len(model_dets) > 0:
                    return model_dets
            except Exception:
                pass

        return self._detect_heuristic(
            arr=arr,
            sensor=sensor,
            source_tile=source_tile,
            acquisition_time=acquisition_time,
            coord_transformer=coord_transformer,
            crs_str=crs_str,
            confidence_heuristic=confidence_heuristic,
        )
