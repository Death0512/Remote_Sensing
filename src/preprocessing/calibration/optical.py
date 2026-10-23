from typing import Optional, Tuple, Union
import numpy as np

from src.domain.enums import Capability, RadiometricState


class OpticalCalibrator:
    capability: Capability = Capability.IMPLEMENTED

    def __init__(self, default_scale: float = 10000.0, default_offset: float = 0.0) -> None:
        self.default_scale = max(default_scale, 1e-6)
        self.default_offset = default_offset

    def dn_to_reflectance(
        self,
        dn: np.ndarray,
        processing_level: str = "L1C",
        scale: Optional[float] = None,
        offset: Optional[float] = None,
        clip_negative: bool = True,
    ) -> Tuple[np.ndarray, RadiometricState]:
        scale_val = scale if scale is not None else self.default_scale
        offset_val = offset if offset is not None else self.default_offset
        if scale_val <= 0:
            raise ValueError("Scale factor must be strictly positive")

        arr = np.asarray(dn, dtype=np.float32)
        reflectance = (arr - offset_val) / scale_val

        if clip_negative:
            reflectance = np.clip(reflectance, 0.0, 1.0)

        level_clean = processing_level.upper().strip()
        if level_clean in ("L2A", "BOA", "SURFACE_REFLECTANCE"):
            state = RadiometricState.SURFACE_REFLECTANCE
        elif level_clean in ("L1C", "TOA", "TOA_REFLECTANCE"):
            state = RadiometricState.TOA_REFLECTANCE
        else:
            state = RadiometricState.TOA_REFLECTANCE

        return reflectance, state


def calibrate_optical_digital_numbers(
    dn: np.ndarray,
    processing_level: str = "L1C",
    scale: Optional[float] = None,
    offset: Optional[float] = None,
    clip_negative: bool = True,
) -> Tuple[np.ndarray, RadiometricState]:
    calibrator = OpticalCalibrator()
    return calibrator.dn_to_reflectance(
        dn=dn,
        processing_level=processing_level,
        scale=scale,
        offset=offset,
        clip_negative=clip_negative,
    )
