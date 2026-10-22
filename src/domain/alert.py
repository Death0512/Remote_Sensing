from dataclasses import dataclass
from src.domain.enums import AlertPriority, ReviewState, Zone
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

    def __post_init__(self) -> None:
        if isinstance(self.zone, str):
            object.__setattr__(self, "zone", Zone(self.zone))
        if isinstance(self.priority, str):
            object.__setattr__(self, "priority", AlertPriority(self.priority))
        if isinstance(self.review_state, str):
            object.__setattr__(self, "review_state", ReviewState(self.review_state))
