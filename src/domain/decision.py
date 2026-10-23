from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Union
from src.domain.enums import GeometryStatus, ReviewAction
from src.domain.provenance import Provenance


@dataclass(frozen=True)
class Decision:
    decision_id: str
    action: Union[ReviewAction, str]
    rationale: str
    timestamp_utc: datetime
    provenance: Provenance
    schema_version: str = "decision.v3"
    target_id: Optional[str] = None
    geometry: Optional[Any] = None
    geometry_status: GeometryStatus = GeometryStatus.UNKNOWN

    def __post_init__(self) -> None:
        if self.provenance is None:
            raise TypeError("provenance is required")
        if isinstance(self.action, str):
            try:
                object.__setattr__(self, "action", ReviewAction(self.action))
            except ValueError:
                pass
        if isinstance(self.geometry_status, str):
            object.__setattr__(self, "geometry_status", GeometryStatus(self.geometry_status))
        if not self.rationale or not self.rationale.strip():
            raise ValueError("rationale must not be empty")
