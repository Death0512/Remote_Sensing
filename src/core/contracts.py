from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class Detection:
    det_id: str
    sensor: Literal["S1_GRD", "S1_RTC", "S2_L1C", "VHR"]
    geom_px: tuple[float, float, float, float]
    geom_wgs84: BaseGeometry
    crs: str
    confidence: float
    length_m: Optional[float]
    heading_deg: Optional[float]
    target_class: Literal[
        "vessel",
        "aircraft_parked",
        "aircraft_inflight",
        "unknown_structure",
        "new_structure",
    ]
    source_tile: str
    acquisition_time: datetime


@dataclass(frozen=True)
class Track:
    track_id: str
    detections: list[Detection]
    velocity_mps: Optional[float]
    heading_deg: Optional[float]
    time_delta_seconds: float


@dataclass(frozen=True)
class Alert:
    alert_id: str
    zone: Literal[1, 2, 3]
    level: Literal["CRITICAL", "HIGH", "STRATEGIC"]
    score_s: float
    is_dark: Optional[bool]
    geojson_feature: dict
    chip_path: str
    audit_status: Literal["PENDING", "CONFIRMED", "DISMISSED"]
