from dataclasses import dataclass
from typing import Any
from src.domain.enums import AlertPriority, GeometryStatus, ReviewState, Zone
from src.domain.provenance import Provenance


@dataclass(frozen=True)
class Alert:
    alert_id: str
    hypothesis_id: str
    zone: Zone
    priority: AlertPriority
    review_state: ReviewState
    bundle_uri: str
    provenance: Provenance
    schema_version: str = "alert.v3"
    geometry_status: GeometryStatus = GeometryStatus.UNKNOWN
    geometry: Any = None

    def __post_init__(self) -> None:
        if self.provenance is None:
            raise TypeError("provenance is required")
        if isinstance(self.zone, str):
            object.__setattr__(self, "zone", Zone(self.zone))
        if isinstance(self.priority, str):
            object.__setattr__(self, "priority", AlertPriority(self.priority))
        if isinstance(self.review_state, str):
            object.__setattr__(self, "review_state", ReviewState(self.review_state))
        if isinstance(self.geometry_status, str):
            object.__setattr__(self, "geometry_status", GeometryStatus(self.geometry_status))
