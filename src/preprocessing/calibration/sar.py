import math
from typing import Optional, Tuple, Union
import numpy as np

from src.domain.enums import Capability, RadiometricState


class SarCalibrator:
    capability: Capability = Capability.IMPLEMENTED

    def __init__(self, eps: float = 1e-10) -> None:
        self.eps = eps

    def calibrate_dn_to_sigma0(
        self,
        dn: np.ndarray,
        lut: Union[float, np.ndarray],
        input_domain: str = "dn",
        to_db: bool = True,
    ) -> Tuple[np.ndarray, RadiometricState]:
        input_mode = input_domain.lower().strip()
        if input_mode == "amplitude":
            raise ValueError("Prohibited mixing: amplitude input is strictly forbidden without explicit squaring to intensity")
        if input_mode not in ("dn", "digital_number", "intensity"):
            raise ValueError(
                f"Invalid input_domain '{input_domain}'. Prohibited mixing of amplitude and intensity; only DN or intensity allowed."
            )

        arr = np.asarray(dn, dtype=np.float32)
        if np.any(np.isnan(arr)):
            raise ValueError("SAR digital numbers contain NaN values")

        lut_arr = np.asarray(lut, dtype=np.float32)
        if np.any(lut_arr <= 0):
            raise ValueError("Calibration LUT values must be strictly positive")

        if input_mode in ("dn", "digital_number"):
            sigma0_linear = (arr ** 2) / (lut_arr ** 2)
        else:
            sigma0_linear = arr / (lut_arr ** 2)

        sigma0_linear = np.maximum(sigma0_linear, self.eps)

        if to_db:
            result = 10.0 * np.log10(sigma0_linear)
        else:
            result = sigma0_linear

        return result, RadiometricState.CALIBRATED_SIGMA0

    def calibrate_dn_to_gamma0(
        self,
        dn: np.ndarray,
        lut: Union[float, np.ndarray],
        incidence_angle_rad: Optional[Union[float, np.ndarray]] = None,
        input_domain: str = "dn",
        to_db: bool = True,
    ) -> Tuple[np.ndarray, RadiometricState]:
        input_mode = input_domain.lower().strip()
        if input_mode == "amplitude":
            raise ValueError("Prohibited mixing: amplitude input is strictly forbidden without explicit squaring to intensity")
        if input_mode not in ("dn", "digital_number", "intensity"):
            raise ValueError(
                f"Invalid input_domain '{input_domain}'. Prohibited mixing of amplitude and intensity; only DN or intensity allowed."
            )

        arr = np.asarray(dn, dtype=np.float32)
        if np.any(np.isnan(arr)):
            raise ValueError("SAR digital numbers contain NaN values")

        lut_arr = np.asarray(lut, dtype=np.float32)
        if np.any(lut_arr <= 0):
            raise ValueError("Calibration LUT values must be strictly positive")

        if incidence_angle_rad is not None:
            inc = np.asarray(incidence_angle_rad, dtype=np.float32)
            cos_inc = np.cos(inc)
            if np.any(cos_inc <= 0.0):
                raise ValueError("Incidence angle must result in positive cos(theta)")
        else:
            cos_inc = 1.0

        if input_mode in ("dn", "digital_number"):
            gamma0_linear = ((arr ** 2) / (lut_arr ** 2)) / cos_inc
        else:
            gamma0_linear = (arr / (lut_arr ** 2)) / cos_inc

        gamma0_linear = np.maximum(gamma0_linear, self.eps)

        if to_db:
            result = 10.0 * np.log10(gamma0_linear)
        else:
            result = gamma0_linear

        return result, RadiometricState.CALIBRATED_GAMMA0


def calibrate_sar_digital_numbers(
    dn: np.ndarray,
    lut: Union[float, np.ndarray],
    target_state: RadiometricState = RadiometricState.CALIBRATED_SIGMA0,
    incidence_angle_rad: Optional[Union[float, np.ndarray]] = None,
    input_domain: str = "dn",
    to_db: bool = True,
) -> Tuple[np.ndarray, RadiometricState]:
    calibrator = SarCalibrator()
    if target_state == RadiometricState.CALIBRATED_SIGMA0:
        return calibrator.calibrate_dn_to_sigma0(
            dn=dn, lut=lut, input_domain=input_domain, to_db=to_db
        )
    if target_state == RadiometricState.CALIBRATED_GAMMA0:
        return calibrator.calibrate_dn_to_gamma0(
            dn=dn,
            lut=lut,
            incidence_angle_rad=incidence_angle_rad,
            input_domain=input_domain,
            to_db=to_db,
        )
    raise ValueError(f"Unsupported SAR target radiometric state: {target_state}")
