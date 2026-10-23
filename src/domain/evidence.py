from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from src.domain.enums import EvidenceKind, GeometryStatus
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
    schema_version: str = "evidence.v3"
    geometry_status: GeometryStatus = GeometryStatus.UNKNOWN
    geometry: Any = None
    source_sensor: Optional[str] = None

    def __post_init__(self) -> None:
        if self.provenance is None:
            raise TypeError("provenance is required")
        if isinstance(self.kind, str):
            object.__setattr__(self, "kind", EvidenceKind(self.kind))
        if isinstance(self.geometry_status, str):
            object.__setattr__(self, "geometry_status", GeometryStatus(self.geometry_status))
        if not isinstance(self.references, tuple):
            object.__setattr__(self, "references", tuple(self.references))
        if self.source_sensor is None and self.source:
            object.__setattr__(self, "source_sensor", self.source)
        elif not self.source and self.source_sensor:
            object.__setattr__(self, "source", self.source_sensor)
