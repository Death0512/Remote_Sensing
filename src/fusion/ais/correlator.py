from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from typing import Optional, Sequence
import uuid
import pyproj
from shapely.geometry import Point

from src.domain.detection import Detection
from src.domain.enums import AisCorrelationState, Capability, EvidenceKind
from src.domain.evidence import Evidence
from src.domain.provenance import Provenance


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


class AisCorrelator:
    capability: Capability = Capability.IMPLEMENTED

    def __init__(
        self,
        max_stale_seconds: float = 3600.0,
        speed_tolerance_knots: float = 5.0,
        heading_tolerance_deg: float = 45.0,
    ) -> None:
        self.max_stale_seconds = max_stale_seconds
        self.speed_tolerance_knots = speed_tolerance_knots
        self.heading_tolerance_deg = heading_tolerance_deg
        self.geod = pyproj.Geod(ellps="WGS84")

    def _ensure_utc(self, dt: datetime) -> datetime:
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _extract_detection_coords(self, detection: Detection) -> tuple[float, float]:
        geom = detection.geom_wgs84
        centroid: Point = geom.centroid
        return float(centroid.x), float(centroid.y)

    def _create_provenance(self, detection: Detection, correlation_id: str) -> Provenance:
        source_id = detection.detection_id
        input_hash = hashlib.sha256(source_id.encode("utf-8")).hexdigest()
        output_hash = hashlib.sha256(correlation_id.encode("utf-8")).hexdigest()
        return Provenance(
            provenance_id=str(uuid.uuid4()),
            source_asset_id=detection.provenance.source_asset_id,
            acquisition_metadata_ref=detection.provenance.acquisition_metadata_ref,
            pipeline_version=detection.provenance.pipeline_version,
            git_commit=detection.provenance.git_commit,
            config_hash=f"ais-correlator-{self.max_stale_seconds}",
            model_id=None,
            model_version=None,
            input_checksum=input_hash,
            output_checksum=output_hash,
            processing_run_id=detection.provenance.processing_run_id,
            processing_time_utc=datetime.now(timezone.utc),
        )

    def correlate(
        self,
        detection: Detection,
        ais_records: Sequence[AisRecord],
        feed_active: bool,
        last_feed_time_utc: Optional[datetime],
        spatial_gate_m: float = 3000.0,
        temporal_gate_s: float = 600.0,
    ) -> Evidence:
        det_time = self._ensure_utc(detection.acquisition_time_utc)
        correlation_id = str(uuid.uuid4())
        prov = self._create_provenance(detection, correlation_id)

        if not feed_active:
            return Evidence(
                evidence_id=f"ev-ais-{correlation_id}",
                kind=EvidenceKind.AIS_CORRELATION,
                value=0.0,
                quality=0.0,
                source="ais_feed",
                timestamp_utc=det_time,
                uncertainty=1.0,
                supports=AisCorrelationState.AIS_UNAVAILABLE.value,
                references=(
                    detection.detection_id,
                    f"state={AisCorrelationState.AIS_UNAVAILABLE.value}",
                ),
                provenance=prov,
            )

        if last_feed_time_utc is not None:
            last_feed_utc = self._ensure_utc(last_feed_time_utc)
            stale_delta = abs((det_time - last_feed_utc).total_seconds())
            if stale_delta > self.max_stale_seconds:
                return Evidence(
                    evidence_id=f"ev-ais-{correlation_id}",
                    kind=EvidenceKind.AIS_CORRELATION,
                    value=0.0,
                    quality=0.2,
                    source="ais_feed",
                    timestamp_utc=det_time,
                    uncertainty=0.8,
                    supports=AisCorrelationState.AIS_STALE.value,
                    references=(
                        detection.detection_id,
                        f"state={AisCorrelationState.AIS_STALE.value}",
                        f"stale_seconds={stale_delta:.1f}",
                    ),
                    provenance=prov,
                )

        det_lon, det_lat = self._extract_detection_coords(detection)

        det_attrs = dict(detection.attributes)
        det_speed = det_attrs.get("speed_knots")
        det_heading = det_attrs.get("heading_deg")

        candidates: list[tuple[float, float, AisRecord]] = []

        for record in ais_records:
            rec_time = self._ensure_utc(record.timestamp)
            dt = abs((det_time - rec_time).total_seconds())
            if dt > temporal_gate_s:
                continue

            _, _, dist = self.geod.inv(det_lon, det_lat, record.lon, record.lat)
            dist = float(dist)

            if dist <= spatial_gate_m:
                candidates.append((dist, dt, record))

        if not candidates:
            return Evidence(
                evidence_id=f"ev-ais-{correlation_id}",
                kind=EvidenceKind.AIS_CORRELATION,
                value=0.0,
                quality=1.0,
                source="ais_correlator",
                timestamp_utc=det_time,
                uncertainty=0.05,
                supports=AisCorrelationState.NO_MATCH.value,
                references=(
                    detection.detection_id,
                    f"state={AisCorrelationState.NO_MATCH.value}",
                ),
                provenance=prov,
            )

        candidates.sort(key=lambda c: c[0])

        if len(candidates) > 1:
            best_dist = candidates[0][0]
            ambiguous_candidates = [c for c in candidates if c[0] <= min(best_dist * 1.5, spatial_gate_m)]
            if len(ambiguous_candidates) > 1:
                mmsis = tuple(c[2].mmsi for c in ambiguous_candidates)
                return Evidence(
                    evidence_id=f"ev-ais-{correlation_id}",
                    kind=EvidenceKind.AIS_CORRELATION,
                    value=0.5,
                    quality=0.5,
                    source="ais_correlator",
                    timestamp_utc=det_time,
                    uncertainty=0.7,
                    supports=AisCorrelationState.CORRELATION_UNCERTAIN.value,
                    references=(
                        detection.detection_id,
                        f"state={AisCorrelationState.CORRELATION_UNCERTAIN.value}",
                        f"candidates_count={len(ambiguous_candidates)}",
                        *tuple(f"candidate_mmsi={m}" for m in mmsis),
                    ),
                    provenance=prov,
                )

        best_dist, best_dt, best_record = candidates[0]

        is_kinematic_mismatch = False
        if det_speed is not None and best_record.speed_knots is not None:
            if abs(float(det_speed) - float(best_record.speed_knots)) > self.speed_tolerance_knots:
                is_kinematic_mismatch = True

        if det_heading is not None and best_record.heading_deg is not None:
            heading_diff = abs(float(det_heading) - float(best_record.heading_deg)) % 360.0
            if heading_diff > 180.0:
                heading_diff = 360.0 - heading_diff
            if heading_diff > self.heading_tolerance_deg:
                is_kinematic_mismatch = True

        if is_kinematic_mismatch:
            state = AisCorrelationState.PARTIAL_MATCH
            val = 0.6
            unc = 0.4
        else:
            state = AisCorrelationState.MATCHED
            val = 1.0
            unc = max(0.01, best_dist / max(spatial_gate_m, 1.0) * 0.2)

        return Evidence(
            evidence_id=f"ev-ais-{correlation_id}",
            kind=EvidenceKind.AIS_CORRELATION,
            value=val,
            quality=0.9,
            source="ais_correlator",
            timestamp_utc=det_time,
            uncertainty=unc,
            supports=state.value,
            references=(
                detection.detection_id,
                f"state={state.value}",
                f"mmsi={best_record.mmsi}",
                f"spatial_delta_m={best_dist:.2f}",
                f"temporal_delta_s={best_dt:.1f}",
            ),
            provenance=prov,
        )
