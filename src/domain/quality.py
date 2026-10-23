from dataclasses import dataclass
from src.domain.enums import QualityTier


@dataclass(frozen=True)
class QualityAssessment:
    tier: QualityTier
    score: float
    flags: tuple[str, ...]
    schema_version: str = "quality.v3"

    def __post_init__(self) -> None:
        if isinstance(self.tier, str):
            object.__setattr__(self, "tier", QualityTier(self.tier))
        if not isinstance(self.flags, tuple):
            object.__setattr__(self, "flags", tuple(self.flags))


__all__ = ["QualityAssessment", "QualityTier"]
