from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple
import pyproj
from shapely.geometry import Point

from src.core.contracts import Detection


@dataclass(frozen=True)
class AisRecord:
    mmsi: str
    lat: float
    lon: float
    timestamp: datetime
    vessel_name: Optional[str] = None
    vessel_type: Optional[str] = None
    speed_knots: Optional[float] = None
    heading_deg: Optional[float] = None


@dataclass(frozen=True)
class CorrelatedDetection:
    detection: Detection
    matched_ais: Optional[AisRecord]
    distance_m: Optional[float]
    time_delta_seconds: Optional[float]
    is_dark: Optional[bool]
    classification: Literal["AIS_MATCHED", "DARK_VESSEL", "AIS_UNCORRELATED_CANDIDATE"]


class AisFusionEngine:
    def __init__(
        self,
        distance_threshold_m: float = 1000.0,
        time_threshold_seconds: float = 1800.0,
        is_nrt_feed_active: bool = False,
    ) -> None:
        self.distance_threshold_m = distance_threshold_m
        self.time_threshold_seconds = time_threshold_seconds
        self.is_nrt_feed_active = is_nrt_feed_active
        self.geod = pyproj.Geod(ellps="WGS84")

    def _extract_detection_coords(self, detection: Detection) -> Tuple[float, float]:
        centroid: Point = detection.geom_wgs84.centroid
        return centroid.x, centroid.y

    def correlate(
        self,
        detection: Detection,
        ais_records: List[AisRecord],
    ) -> CorrelatedDetection:
        det_lon, det_lat = self._extract_detection_coords(detection)
        det_time = detection.acquisition_time
        if det_time.tzinfo is None or det_time.tzinfo.utcoffset(det_time) is None:
            det_time = det_time.replace(tzinfo=timezone.utc)

        best_record: Optional[AisRecord] = None
        min_dist: Optional[float] = None
        min_dt: Optional[float] = None

        for rec in ais_records:
            rec_time = rec.timestamp
            if rec_time.tzinfo is None or rec_time.tzinfo.utcoffset(rec_time) is None:
                rec_time = rec_time.replace(tzinfo=timezone.utc)
            dt = abs((det_time - rec_time).total_seconds())
            if dt > self.time_threshold_seconds:
                continue

            _, _, dist = self.geod.inv(det_lon, det_lat, rec.lon, rec.lat)
            dist = float(dist)

            if dist <= self.distance_threshold_m:
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    min_dt = dt
                    best_record = rec

        if best_record is not None:
            return CorrelatedDetection(
                detection=detection,
                matched_ais=best_record,
                distance_m=min_dist,
                time_delta_seconds=min_dt,
                is_dark=False,
                classification="AIS_MATCHED",
            )

        if self.is_nrt_feed_active:
            return CorrelatedDetection(
                detection=detection,
                matched_ais=None,
                distance_m=None,
                time_delta_seconds=None,
                is_dark=True,
                classification="DARK_VESSEL",
            )

        return CorrelatedDetection(
            detection=detection,
            matched_ais=None,
            distance_m=None,
            time_delta_seconds=None,
            is_dark=None,
            classification="AIS_UNCORRELATED_CANDIDATE",
        )

    def correlate_batch(
        self,
        detections: List[Detection],
        ais_records: List[AisRecord],
    ) -> List[CorrelatedDetection]:
        return [self.correlate(d, ais_records) for d in detections]
