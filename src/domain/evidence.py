from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from src.domain.enums import EvidenceKind
from src.domain.provenance import Provenance


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    kind: EvidenceKind
    value: float
    quality: float
    source: str
    timestamp_utc: datetime
    uncertainty: Optional[float]
    supports: str
    references: tuple[str, ...]
    provenance: Provenance

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            object.__setattr__(self, "kind", EvidenceKind(self.kind))
        if not isinstance(self.references, tuple):
            object.__setattr__(self, "references", tuple(self.references))
