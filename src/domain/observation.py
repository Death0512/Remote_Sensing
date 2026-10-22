from dataclasses import dataclass
from datetime import datetime
from shapely.geometry.base import BaseGeometry
from src.domain.enums import GeometricState, QualityTier, RadiometricState, SensorType
from src.domain.provenance import Provenance


@dataclass(frozen=True)
class Observation:
    observation_id: str
    source: str
    platform: str
    sensor: SensorType
    product: str
    processing_level: str
    acquisition_time_utc: datetime
    crs: str
    ground_sample_distance_m: float
    bounds_wgs84: BaseGeometry
    asset_uri: str
    metadata_uri: str
    quality_flags: tuple[str, ...]
    provenance: Provenance

    def __post_init__(self) -> None:
        if isinstance(self.sensor, str):
            object.__setattr__(self, "sensor", SensorType(self.sensor))
        if not isinstance(self.quality_flags, tuple):
            object.__setattr__(self, "quality_flags", tuple(self.quality_flags))


@dataclass(frozen=True)
class ProcessedObservation:
    processed_id: str
    observation_id: str
    preprocessing_profile: str
    radiometric_state: RadiometricState
    geometric_state: GeometricState
    masks: tuple[tuple[str, str], ...]
    quality_tier: QualityTier
    provenance: Provenance

    def __post_init__(self) -> None:
        if isinstance(self.radiometric_state, str):
            object.__setattr__(self, "radiometric_state", RadiometricState(self.radiometric_state))
        if isinstance(self.geometric_state, str):
            object.__setattr__(self, "geometric_state", GeometricState(self.geometric_state))
        if isinstance(self.quality_tier, str):
            object.__setattr__(self, "quality_tier", QualityTier(self.quality_tier))
        if isinstance(self.masks, dict):
            object.__setattr__(self, "masks", tuple(self.masks.items()))
        elif not isinstance(self.masks, tuple):
            object.__setattr__(self, "masks", tuple(self.masks))
