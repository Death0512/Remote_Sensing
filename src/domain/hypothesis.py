from dataclasses import dataclass
from typing import Any, Optional
from src.domain.enums import CalibrationState, GeometryStatus, HypothesisStatement, ScoreMethod
from src.domain.provenance import Provenance


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    statement: HypothesisStatement
    evidence_ids: tuple[str, ...]
    supporting_evidence: tuple[Any, ...]
    refuting_evidence: tuple[Any, ...]
    score: Optional[float]
    score_method: ScoreMethod
    calibration_state: CalibrationState
    explanation: str
    provenance: Provenance
    primary_explanation: Optional[str] = None
    detailed_rationale: Optional[str] = None
    schema_version: str = "hypothesis.v3"
    geometry_status: GeometryStatus = GeometryStatus.UNKNOWN
    geometry: Any = None
    raw_score: Optional[float] = None
    calibrated_probability: Optional[float] = None
    uncertainty: Optional[float] = None

    def __post_init__(self) -> None:
        if self.provenance is None:
            raise TypeError("provenance is required")
        if isinstance(self.statement, str):
            object.__setattr__(self, "statement", HypothesisStatement(self.statement))
        if isinstance(self.score_method, str):
            object.__setattr__(self, "score_method", ScoreMethod(self.score_method))
        if isinstance(self.calibration_state, str):
            object.__setattr__(self, "calibration_state", CalibrationState(self.calibration_state))
        if isinstance(self.geometry_status, str):
            object.__setattr__(self, "geometry_status", GeometryStatus(self.geometry_status))
        if not isinstance(self.evidence_ids, tuple):
            object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        if not isinstance(self.supporting_evidence, tuple):
            object.__setattr__(self, "supporting_evidence", tuple(self.supporting_evidence))
        if not isinstance(self.refuting_evidence, tuple):
            object.__setattr__(self, "refuting_evidence", tuple(self.refuting_evidence))
        if not self.evidence_ids and self.supporting_evidence:
            object.__setattr__(
                self,
                "evidence_ids",
                tuple(
                    str(getattr(e, "evidence_id", e)) for e in self.supporting_evidence
                ),
            )
        if not self.explanation or not self.explanation.strip():
            raise ValueError("Hypothesis requires non-empty explanation")
        if not self.evidence_ids:
            raise ValueError("Hypothesis requires at least one evidence item")
        if self.raw_score is None and self.score is not None:
            object.__setattr__(self, "raw_score", self.score)
        elif self.score is None and self.raw_score is not None:
            object.__setattr__(self, "score", self.raw_score)
        if self.calibration_state == CalibrationState.NOT_CALIBRATED and self.calibrated_probability is not None:
            raise ValueError("calibrated_probability must be None when calibration_state is NOT_CALIBRATED")
        if self.primary_explanation is None:
            object.__setattr__(self, "primary_explanation", self.explanation)
        if self.detailed_rationale is None:
            object.__setattr__(self, "detailed_rationale", self.explanation)
