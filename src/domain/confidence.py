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
    schema_version: str = "confidence_vector.v3"

    def __post_init__(self) -> None:
        if isinstance(self.calibration_state, str):
            object.__setattr__(self, "calibration_state", CalibrationState(self.calibration_state))
        if self.calibration_state == CalibrationState.NOT_CALIBRATED and self.calibrated_probability is not None:
            raise ValueError("calibrated_probability must be None when calibration_state is NOT_CALIBRATED")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            return self.raw_score == float(other)
        if isinstance(other, ConfidenceVector):
            return self.raw_score == other.raw_score
        return super().__eq__(other)

    def __gt__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            return self.raw_score > float(other)
        if isinstance(other, ConfidenceVector):
            return self.raw_score > other.raw_score
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            return self.raw_score >= float(other)
        if isinstance(other, ConfidenceVector):
            return self.raw_score >= other.raw_score
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            return self.raw_score < float(other)
        if isinstance(other, ConfidenceVector):
            return self.raw_score < other.raw_score
        return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            return self.raw_score <= float(other)
        if isinstance(other, ConfidenceVector):
            return self.raw_score <= other.raw_score
        return NotImplemented

    def __float__(self) -> float:
        return float(self.raw_score)

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_score": self.raw_score,
            "model_score": self.model_score,
            "calibrated_probability": self.calibrated_probability,
            "uncertainty": self.uncertainty,
            "quality_score": self.quality_score,
            "calibration_state": self.calibration_state.value if hasattr(self.calibration_state, "value") else str(self.calibration_state),
            "schema_version": self.schema_version,
        }
