from dataclasses import dataclass
from typing import Optional
from src.domain.enums import CalibrationState


@dataclass(frozen=True)
class ConfidenceVector:
    raw_score: float
    model_score: Optional[float] = None
    calibrated_probability: Optional[float] = None
    uncertainty: Optional[float] = None
    quality_score: float = 1.0
    calibration_state: CalibrationState = CalibrationState.NOT_CALIBRATED

    def __post_init__(self) -> None:
        if isinstance(self.calibration_state, str):
            object.__setattr__(self, "calibration_state", CalibrationState(self.calibration_state))
        if self.calibration_state == CalibrationState.NOT_CALIBRATED and self.calibrated_probability is not None:
            raise ValueError("calibrated_probability must be None when calibration_state is NOT_CALIBRATED")
