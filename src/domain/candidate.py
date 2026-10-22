from dataclasses import dataclass
from shapely.geometry.base import BaseGeometry
from src.domain.enums import Capability
from src.domain.provenance import Provenance


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    processed_id: str
    geom_px: tuple[float, float, float, float]
    geom_wgs84: BaseGeometry
    generator: str
    generator_capability: Capability
    features: tuple[tuple[str, float], ...]
    quality_flags: tuple[str, ...]
    provenance: Provenance

    def __post_init__(self) -> None:
        if isinstance(self.generator_capability, str):
            object.__setattr__(self, "generator_capability", Capability(self.generator_capability))
        if not isinstance(self.geom_px, tuple):
            object.__setattr__(self, "geom_px", tuple(self.geom_px))
        if isinstance(self.features, dict):
            object.__setattr__(self, "features", tuple(self.features.items()))
        elif not isinstance(self.features, tuple):
            object.__setattr__(self, "features", tuple(self.features))
        if not isinstance(self.quality_flags, tuple):
            object.__setattr__(self, "quality_flags", tuple(self.quality_flags))
