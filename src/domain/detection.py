from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from shapely.geometry.base import BaseGeometry
from src.domain.confidence import ConfidenceVector
from src.domain.enums import GeometryStatus, SensorType, TargetClass
from src.domain.provenance import Provenance


@dataclass(frozen=True)
class Detection:
    detection_id: str = ""
    candidate_id: str = ""
    sensor: Any = SensorType.S1_SAR
    geom_wgs84: Any = None
    geom_px: Any = (0.0, 0.0, 0.0, 0.0)
    crs: str = "EPSG:4326"
    target_class: Any = TargetClass.VESSEL
    confidence: Any = None
    attributes: tuple[tuple[str, Any], ...] = ()
    acquisition_time_utc: Optional[datetime] = None
    provenance: Optional[Provenance] = None
    schema_version: str = "detection.v3"
    geometry_status: GeometryStatus = GeometryStatus.GEOCODED
    geometry: Any = None
    det_id: Optional[str] = None
    acquisition_time: Optional[datetime] = None
    length_m: Optional[float] = None
    heading_deg: Optional[float] = None
    source_tile: Optional[str] = None
    bounding_box: Optional[Any] = None

    def __post_init__(self) -> None:
        is_legacy = self.source_tile is not None and not self.candidate_id

        if not self.detection_id and self.det_id:
            object.__setattr__(self, "detection_id", self.det_id)
        elif self.detection_id and not self.det_id:
            object.__setattr__(self, "det_id", self.detection_id)

        if not self.detection_id:
            raise TypeError("detection_id or det_id is required")

        if self.acquisition_time_utc is None and self.acquisition_time is not None:
            object.__setattr__(self, "acquisition_time_utc", self.acquisition_time)
        elif self.acquisition_time_utc is not None and self.acquisition_time is None:
            object.__setattr__(self, "acquisition_time", self.acquisition_time_utc)

        if self.provenance is None:
            if is_legacy:
                object.__setattr__(
                    self,
                    "provenance",
                    Provenance(
                        provenance_id="prov-legacy",
                        source_asset_id=self.source_tile or "legacy",
                        acquisition_metadata_ref="legacy",
                        pipeline_version="legacy",
                        git_commit="legacy",
                        config_hash="legacy",
                        model_id=None,
                        model_version=None,
                        input_checksum="legacy",
                        output_checksum="legacy",
                        processing_run_id="legacy",
                        processing_time_utc=self.acquisition_time_utc or datetime.now(timezone.utc),
                    ),
                )
            else:
                raise TypeError("provenance is required")

        if isinstance(self.sensor, str) and not is_legacy:
            try:
                object.__setattr__(self, "sensor", SensorType(self.sensor))
            except ValueError:
                pass

        if isinstance(self.target_class, str) and not is_legacy:
            try:
                object.__setattr__(self, "target_class", TargetClass(self.target_class))
            except ValueError:
                pass

        if isinstance(self.confidence, (int, float)):
            object.__setattr__(self, "confidence", ConfidenceVector(raw_score=float(self.confidence)))
        elif self.confidence is None:
            object.__setattr__(self, "confidence", ConfidenceVector(raw_score=1.0))

        if hasattr(self.geom_px, "to_tuple"):
            if self.bounding_box is None:
                object.__setattr__(self, "bounding_box", self.geom_px)
            object.__setattr__(self, "geom_px", self.geom_px.to_tuple())
        elif not isinstance(self.geom_px, tuple):
            object.__setattr__(self, "geom_px", tuple(self.geom_px))

        attrs = dict(self.attributes) if isinstance(self.attributes, (tuple, list, dict)) else {}
        if self.length_m is not None and "length_m" not in attrs:
            attrs["length_m"] = self.length_m
        if self.heading_deg is not None and "heading_deg" not in attrs:
            attrs["heading_deg"] = self.heading_deg
        if self.source_tile is not None and "source_tile" not in attrs:
            attrs["source_tile"] = self.source_tile
        if self.length_m is None and "length_m" in attrs:
            object.__setattr__(self, "length_m", attrs["length_m"])
        if self.heading_deg is None and "heading_deg" in attrs:
            object.__setattr__(self, "heading_deg", attrs["heading_deg"])
        if self.source_tile is None and "source_tile" in attrs:
            object.__setattr__(self, "source_tile", attrs["source_tile"])
        object.__setattr__(self, "attributes", tuple(attrs.items()))

        if isinstance(self.geometry_status, str):
            object.__setattr__(self, "geometry_status", GeometryStatus(self.geometry_status))

        if self.geometry is None and self.geom_wgs84 is not None:
            object.__setattr__(self, "geometry", self.geom_wgs84)
        elif self.geom_wgs84 is None and self.geometry is not None:
            object.__setattr__(self, "geom_wgs84", self.geometry)
