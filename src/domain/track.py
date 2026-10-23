from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from src.domain.enums import GeometryStatus, MotionClass, TrackLifecycle
from src.domain.provenance import Provenance


@dataclass(frozen=True)
class Track:
    track_id: str
    detection_ids: tuple[str, ...]
    state_estimates: tuple[tuple[tuple[str, float], ...], ...]
    track_confidence: float
    lifecycle: TrackLifecycle
    provenance: Provenance
    schema_version: str = "track.v3"
    geometry_status: GeometryStatus = GeometryStatus.UNKNOWN
    geometry: Any = None
    motion_class: MotionClass = MotionClass.UNKNOWN
    state: tuple[float, ...] = ()
    state_covariance: tuple[tuple[float, ...], ...] = ()
    last_update_utc: Optional[datetime] = None
    prediction_time_utc: Optional[datetime] = None
    associated_detection_ids: tuple[str, ...] = ()
    sensor_sources: tuple[str, ...] = ()
    quality: float = 1.0

    def __post_init__(self) -> None:
        if self.provenance is None:
            raise TypeError("provenance is required")
        if isinstance(self.lifecycle, str):
            object.__setattr__(self, "lifecycle", TrackLifecycle(self.lifecycle))
        if isinstance(self.motion_class, str):
            object.__setattr__(self, "motion_class", MotionClass(self.motion_class))
        if isinstance(self.geometry_status, str):
            object.__setattr__(self, "geometry_status", GeometryStatus(self.geometry_status))
        if not isinstance(self.detection_ids, tuple):
            object.__setattr__(self, "detection_ids", tuple(self.detection_ids))
        if not isinstance(self.associated_detection_ids, tuple):
            object.__setattr__(self, "associated_detection_ids", tuple(self.associated_detection_ids))
        if not self.associated_detection_ids and self.detection_ids:
            object.__setattr__(self, "associated_detection_ids", self.detection_ids)
        elif not self.detection_ids and self.associated_detection_ids:
            object.__setattr__(self, "detection_ids", self.associated_detection_ids)
        if not isinstance(self.sensor_sources, tuple):
            object.__setattr__(self, "sensor_sources", tuple(self.sensor_sources))
        if not isinstance(self.state, tuple):
            object.__setattr__(self, "state", tuple(self.state))
        if not isinstance(self.state_covariance, tuple):
            object.__setattr__(self, "state_covariance", tuple(self.state_covariance))
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
