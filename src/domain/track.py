from dataclasses import dataclass
from src.domain.enums import TrackLifecycle
from src.domain.provenance import Provenance


@dataclass(frozen=True)
class Track:
    track_id: str
    detection_ids: tuple[str, ...]
    state_estimates: tuple[tuple[tuple[str, float], ...], ...]
    track_confidence: float
    lifecycle: TrackLifecycle
    provenance: Provenance

    def __post_init__(self) -> None:
        if isinstance(self.lifecycle, str):
            object.__setattr__(self, "lifecycle", TrackLifecycle(self.lifecycle))
        if not isinstance(self.detection_ids, tuple):
            object.__setattr__(self, "detection_ids", tuple(self.detection_ids))
        if self.state_estimates:
            converted = []
            for item in self.state_estimates:
                if isinstance(item, dict):
                    converted.append(tuple(item.items()))
                elif isinstance(item, (list, tuple)):
                    converted.append(tuple(item))
            object.__setattr__(self, "state_estimates", tuple(converted))
        elif not isinstance(self.state_estimates, tuple):
            object.__setattr__(self, "state_estimates", tuple(self.state_estimates))
