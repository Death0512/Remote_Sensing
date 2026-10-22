from dataclasses import dataclass
from datetime import datetime
from typing import Any
from shapely.geometry.base import BaseGeometry
from src.domain.confidence import ConfidenceVector
from src.domain.enums import SensorType, TargetClass
from src.domain.provenance import Provenance


@dataclass(frozen=True)
class Detection:
    detection_id: str
    candidate_id: str
    sensor: SensorType
    geom_wgs84: BaseGeometry
    geom_px: tuple[float, float, float, float]
    crs: str
    target_class: TargetClass
    confidence: ConfidenceVector
    attributes: tuple[tuple[str, Any], ...]
    acquisition_time_utc: datetime
    provenance: Provenance

    def __post_init__(self) -> None:
        if isinstance(self.sensor, str):
            object.__setattr__(self, "sensor", SensorType(self.sensor))
        if isinstance(self.target_class, str):
            object.__setattr__(self, "target_class", TargetClass(self.target_class))
        if not isinstance(self.geom_px, tuple):
            object.__setattr__(self, "geom_px", tuple(self.geom_px))
        if isinstance(self.attributes, dict):
            object.__setattr__(self, "attributes", tuple(self.attributes.items()))
        elif not isinstance(self.attributes, tuple):
            object.__setattr__(self, "attributes", tuple(self.attributes))
