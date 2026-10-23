from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from src.domain.enums import GeometryStatus
from src.domain.provenance import Provenance


@dataclass(frozen=True)
class Association:
    association_id: str
    source_id: str
    target_id: str
    association_type: str
    residual: float
    gate_statistic: float
    confidence: float
    provenance: Provenance
    schema_version: str = "association.v3"
    surviving_uncertainty: Optional[float] = None
    quality_flags: tuple[str, ...] = ()
    geometry: Optional[Any] = None
    geometry_status: GeometryStatus = GeometryStatus.UNKNOWN
    timestamp_utc: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.provenance is None:
            raise TypeError("provenance is required")
        if isinstance(self.geometry_status, str):
            object.__setattr__(self, "geometry_status", GeometryStatus(self.geometry_status))
        if not isinstance(self.quality_flags, tuple):
            object.__setattr__(self, "quality_flags", tuple(self.quality_flags))
