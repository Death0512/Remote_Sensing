from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Union
from src.domain.enums import CalibrationState, GeometryStatus, SensorType
from src.domain.geometry import GeodeticGeometry, MapGeometry, PixelGeometry
from src.domain.provenance import Provenance


@dataclass(frozen=True)
class Measurement:
    measurement_id: str
    source_observation_id: str
    sensor_type: SensorType
    value: float
    unit: str
    calibration_status: CalibrationState
    quality_flags: tuple[str, ...]
    provenance: Provenance
    schema_version: str = "measurement.v3"
    geometry: Optional[Union[PixelGeometry, MapGeometry, GeodeticGeometry, Any]] = None
    geometry_status: GeometryStatus = GeometryStatus.UNKNOWN
    timestamp_utc: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.provenance is None:
            raise TypeError("provenance is required")
        if isinstance(self.sensor_type, str):
            object.__setattr__(self, "sensor_type", SensorType(self.sensor_type))
        if isinstance(self.calibration_status, str):
            object.__setattr__(self, "calibration_status", CalibrationState(self.calibration_status))
        if isinstance(self.geometry_status, str):
            object.__setattr__(self, "geometry_status", GeometryStatus(self.geometry_status))
        if not isinstance(self.quality_flags, tuple):
            object.__setattr__(self, "quality_flags", tuple(self.quality_flags))

    @property
    def sensor(self) -> SensorType:
        return self.sensor_type

    @property
    def observation_id(self) -> str:
        return self.source_observation_id
