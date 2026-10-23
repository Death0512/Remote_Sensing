from dataclasses import dataclass
from typing import Any, Optional
from shapely.geometry.base import BaseGeometry
from src.domain.enums import Capability, GeometryStatus
from src.domain.provenance import Provenance


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    processed_id: str
    geom_px: Any
    geom_wgs84: BaseGeometry
    generator: str
    generator_capability: Capability
    features: tuple[tuple[str, float], ...]
    quality_flags: tuple[str, ...]
    provenance: Provenance
    schema_version: str = "candidate.v3"
    geometry_status: GeometryStatus = GeometryStatus.GEOCODED
    geometry: Any = None
    bounding_box: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.provenance is None:
            raise TypeError("provenance is required")
        if isinstance(self.generator_capability, str):
            object.__setattr__(self, "generator_capability", Capability(self.generator_capability))
        if isinstance(self.geometry_status, str):
            object.__setattr__(self, "geometry_status", GeometryStatus(self.geometry_status))
        if hasattr(self.geom_px, "to_tuple"):
            if self.bounding_box is None:
                object.__setattr__(self, "bounding_box", self.geom_px)
            object.__setattr__(self, "geom_px", self.geom_px.to_tuple())
        elif not isinstance(self.geom_px, tuple):
            object.__setattr__(self, "geom_px", tuple(self.geom_px))
        if isinstance(self.features, dict):
            object.__setattr__(self, "features", tuple(self.features.items()))
        elif not isinstance(self.features, tuple):
            object.__setattr__(self, "features", tuple(self.features))
        if not isinstance(self.quality_flags, tuple):
            object.__setattr__(self, "quality_flags", tuple(self.quality_flags))
        if self.geometry is None and self.geom_wgs84 is not None:
            object.__setattr__(self, "geometry", self.geom_wgs84)
