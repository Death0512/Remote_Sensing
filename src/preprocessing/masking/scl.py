from enum import IntEnum
from typing import Dict, Optional, Set, Tuple
import numpy as np

from src.domain.enums import Capability, QualityTier


class SclClass(IntEnum):
    NO_DATA = 0
    SATURATED_OR_DEFECTIVE = 1
    DARK_AREA_PIXELS = 2
    CLOUD_SHADOWS = 3
    VEGETATION = 4
    NOT_VEGETATED = 5
    WATER = 6
    UNCLASSIFIED = 7
    CLOUD_MEDIUM_PROBABILITY = 8
    CLOUD_HIGH_PROBABILITY = 9
    THIN_CIRRUS = 10
    SNOW = 11


DEFAULT_CLOUD_CLASSES: Set[int] = {
    int(SclClass.CLOUD_MEDIUM_PROBABILITY),
    int(SclClass.CLOUD_HIGH_PROBABILITY),
    int(SclClass.THIN_CIRRUS),
}

DEFAULT_SHADOW_CLASSES: Set[int] = {
    int(SclClass.CLOUD_SHADOWS),
}

DEFAULT_CONTAMINATION_CLASSES: Set[int] = {
    int(SclClass.CLOUD_SHADOWS),
    int(SclClass.CLOUD_MEDIUM_PROBABILITY),
    int(SclClass.CLOUD_HIGH_PROBABILITY),
    int(SclClass.THIN_CIRRUS),
}


class SclMaskExtractor:
    capability: Capability = Capability.IMPLEMENTED

    def __init__(self, cloud_threshold: float = 0.20) -> None:
        self.cloud_threshold = cloud_threshold

    def extract_masks(
        self,
        scl: np.ndarray,
        cloud_classes: Optional[Set[int]] = None,
        shadow_classes: Optional[Set[int]] = None,
    ) -> Dict[str, np.ndarray]:
        scl_arr = np.asarray(scl, dtype=np.uint8)
        c_classes = cloud_classes if cloud_classes is not None else DEFAULT_CLOUD_CLASSES
        s_classes = shadow_classes if shadow_classes is not None else DEFAULT_SHADOW_CLASSES

        cloud_mask = np.isin(scl_arr, list(c_classes))
        shadow_mask = np.isin(scl_arr, list(s_classes))
        cirrus_mask = scl_arr == int(SclClass.THIN_CIRRUS)
        valid_mask = scl_arr != int(SclClass.NO_DATA)
        invalid_mask = ~valid_mask | cloud_mask | shadow_mask

        return {
            "cloud_mask": cloud_mask,
            "shadow_mask": shadow_mask,
            "cirrus_mask": cirrus_mask,
            "valid_mask": valid_mask,
            "contaminated_mask": invalid_mask,
        }

    def evaluate_quality(
        self,
        scl: np.ndarray,
        cloud_threshold: Optional[float] = None,
        include_shadows: bool = True,
    ) -> Tuple[QualityTier, float, Dict[str, float]]:
        scl_arr = np.asarray(scl, dtype=np.uint8)
        thresh = cloud_threshold if cloud_threshold is not None else self.cloud_threshold

        valid_pixels = np.sum(scl_arr != int(SclClass.NO_DATA))
        total_pixels = scl_arr.size

        if total_pixels == 0 or valid_pixels == 0:
            return QualityTier.OUTSIDE_VALID_FOOTPRINT, 1.0, {"cloud_ratio": 1.0, "shadow_ratio": 0.0}

        masks = self.extract_masks(scl_arr)
        cloud_count = int(np.sum(masks["cloud_mask"]))
        shadow_count = int(np.sum(masks["shadow_mask"]))
        cirrus_count = int(np.sum(masks["cirrus_mask"]))

        cloud_ratio = float(cloud_count) / float(valid_pixels)
        shadow_ratio = float(shadow_count) / float(valid_pixels)
        cirrus_ratio = float(cirrus_count) / float(valid_pixels)

        contamination_ratio = float(cloud_count + (shadow_count if include_shadows else 0)) / float(valid_pixels)

        stats = {
            "cloud_ratio": cloud_ratio,
            "shadow_ratio": shadow_ratio,
            "cirrus_ratio": cirrus_ratio,
            "contamination_ratio": contamination_ratio,
        }

        if contamination_ratio > thresh:
            return QualityTier.CLOUD_CONTAMINATED, contamination_ratio, stats
        return QualityTier.DATA_OK, contamination_ratio, stats


def extract_scl_masks(
    scl: np.ndarray,
    cloud_classes: Optional[Set[int]] = None,
    shadow_classes: Optional[Set[int]] = None,
) -> Dict[str, np.ndarray]:
    extractor = SclMaskExtractor()
    return extractor.extract_masks(scl, cloud_classes=cloud_classes, shadow_classes=shadow_classes)


def evaluate_scl_cloud_coverage(
    scl: np.ndarray,
    cloud_threshold: float = 0.20,
    include_shadows: bool = True,
) -> Tuple[QualityTier, float, Dict[str, float]]:
    extractor = SclMaskExtractor(cloud_threshold=cloud_threshold)
    return extractor.evaluate_quality(scl, cloud_threshold=cloud_threshold, include_shadows=include_shadows)
