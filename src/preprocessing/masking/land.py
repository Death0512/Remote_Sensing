from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
from shapely.geometry.base import BaseGeometry

from src.domain.enums import Capability
from src.preprocessing.sar.preprocessor import SarPreprocessor

try:
    from rasterio.features import rasterize
    from rasterio.transform import Affine
except ImportError:
    rasterize = None
    Affine = None


class LandMaskExtractor:
    capability: Capability = Capability.IMPLEMENTED

    def __init__(self, buffer_pixels: int = 2) -> None:
        self.buffer_pixels = max(0, int(buffer_pixels))
        self.sar_preprocessor = SarPreprocessor(buffer_pixels=self.buffer_pixels)

    def buffer_mask(
        self, land_mask: np.ndarray, buffer_pixels: Optional[int] = None
    ) -> np.ndarray:
        buf = self.buffer_pixels if buffer_pixels is None else max(0, int(buffer_pixels))
        if buf == self.buffer_pixels:
            return self.sar_preprocessor.buffer_land_mask(land_mask)
        return SarPreprocessor(buffer_pixels=buf).buffer_land_mask(land_mask)

    def rasterize_polygons(
        self,
        polygons: Any,
        shape: Tuple[int, int],
        transform: Any,
        all_touched: bool = False,
        buffer_pixels: Optional[int] = None,
    ) -> np.ndarray:
        if rasterize is None:
            return np.zeros(shape, dtype=bool)

        aff = transform
        if Affine is not None:
            if isinstance(transform, Affine):
                aff = transform
            elif hasattr(transform, "affine"):
                aff = transform.affine
            elif isinstance(transform, (tuple, list)) and len(transform) == 6:
                aff = Affine(*transform)

        if isinstance(polygons, (BaseGeometry, dict)):
            shapes = [polygons]
        elif isinstance(polygons, Iterable):
            shapes = list(polygons)
        else:
            shapes = [polygons]

        mask = rasterize(
            shapes=shapes,
            out_shape=shape,
            transform=aff,
            fill=0,
            default_value=1,
            all_touched=all_touched,
            dtype=np.uint8,
        ).astype(bool)

        buf = self.buffer_pixels if buffer_pixels is None else max(0, int(buffer_pixels))
        if buf > 0:
            mask = self.buffer_mask(mask, buffer_pixels=buf)
        return mask

    def extract_land_mask(
        self,
        shape: Tuple[int, int],
        land_mask: Optional[np.ndarray] = None,
        polygons: Optional[Any] = None,
        transform: Optional[Any] = None,
        buffer_pixels: Optional[int] = None,
        all_touched: bool = False,
    ) -> np.ndarray:
        if land_mask is not None:
            buf = self.buffer_pixels if buffer_pixels is None else max(0, int(buffer_pixels))
            if buf > 0:
                return self.buffer_mask(land_mask, buffer_pixels=buf)
            return land_mask.astype(bool)

        if polygons is not None and transform is not None:
            return self.rasterize_polygons(
                polygons=polygons,
                shape=shape,
                transform=transform,
                all_touched=all_touched,
                buffer_pixels=buffer_pixels,
            )

        return np.zeros(shape, dtype=bool)

    def apply_land_mask(
        self,
        intensity: np.ndarray,
        land_mask: np.ndarray,
        fill_value: float = 0.0,
    ) -> np.ndarray:
        return self.sar_preprocessor.apply_land_mask(
            intensity=intensity,
            land_mask=land_mask,
            fill_value=fill_value,
        )


def extract_land_mask(
    shape: Tuple[int, int],
    land_mask: Optional[np.ndarray] = None,
    polygons: Optional[Any] = None,
    transform: Optional[Any] = None,
    buffer_pixels: int = 2,
    all_touched: bool = False,
) -> np.ndarray:
    extractor = LandMaskExtractor(buffer_pixels=buffer_pixels)
    return extractor.extract_land_mask(
        shape=shape,
        land_mask=land_mask,
        polygons=polygons,
        transform=transform,
        buffer_pixels=buffer_pixels,
        all_touched=all_touched,
    )
