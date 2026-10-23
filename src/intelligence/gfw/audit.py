from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from typing import Any, Mapping, Optional
import uuid

from src.domain.detection import Detection
from src.domain.enums import AlertPriority, Capability, EvidenceKind
from src.domain.evidence import Evidence
from src.domain.provenance import Provenance
from src.provenance.engine import compute_bytes_checksum


@dataclass(frozen=True)
class GfwAuditReport:
    report_id: str
    detection_id: str
    mmsi: Optional[str]
    vessel_name: Optional[str]
    event_timestamp_utc: datetime
    latency_hours: float
    is_retrospective: bool
    evidence: Evidence
    provenance: Provenance
    spatial_summary: str
    confidence: float


class GfwRetrospectiveAuditor:
    capability: Capability = Capability.IMPLEMENTED

    def __init__(
        self,
        latency_threshold_hours: float = 2.0,
        pipeline_version: str = "3.0.0",
    ) -> None:
        self.latency_threshold_hours = latency_threshold_hours
        self.pipeline_version = pipeline_version

    def _ensure_utc(self, dt: datetime) -> datetime:
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def generate_evidence(
        self,
        gfw_record: Mapping[str, Any],
        detection_id: str,
        provenance: Optional[Provenance] = None,
        alert_priority: Optional[AlertPriority] = None,
    ) -> Evidence:
        if alert_priority == AlertPriority.CRITICAL:
            raise ValueError(
                "GFW retrospective data cannot trigger real-time critical priority alerts"
            )

        record_id = str(
            gfw_record.get("mmsi")
            or gfw_record.get("event_id")
            or uuid.uuid4()
        )
        raw_ts = gfw_record.get("timestamp")
        if isinstance(raw_ts, str):
            event_time = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        elif isinstance(raw_ts, datetime):
            event_time = raw_ts
        else:
            event_time = datetime.now(timezone.utc)
        event_time = self._ensure_utc(event_time)

        now_utc = datetime.now(timezone.utc)
        latency_seconds = max(0.0, (now_utc - event_time).total_seconds())
        latency_hours = latency_seconds / 3600.0

        evidence_id = f"ev-gfw-{uuid.uuid4().hex[:12]}"
        input_data = f"{detection_id}:{record_id}".encode("utf-8")
        output_data = evidence_id.encode("utf-8")

        prov = Provenance(
            provenance_id=str(uuid.uuid4()),
            source_asset_id=provenance.source_asset_id if provenance else "GFW_CATALOG",
            acquisition_metadata_ref=provenance.acquisition_metadata_ref if provenance else "s3://gfw/events",
            pipeline_version=self.pipeline_version,
            git_commit=provenance.git_commit if provenance else "fedcba9",
            config_hash=compute_bytes_checksum(
                f"gfw-audit-threshold={self.latency_threshold_hours}".encode("utf-8")
            ),
            model_id=None,
            model_version=None,
            input_checksum=compute_bytes_checksum(input_data),
            output_checksum=compute_bytes_checksum(output_data),
            processing_run_id=provenance.processing_run_id if provenance else f"run-gfw-{uuid.uuid4().hex[:8]}",
            processing_time_utc=now_utc,
            parent_provenance_id=provenance.provenance_id if provenance else None,
        )

        confidence = float(gfw_record.get("confidence", 0.8))
        vessel_name = str(gfw_record.get("vessel_name", "UNKNOWN"))
        event_type = str(gfw_record.get("event_type", "FISHING_EVENT"))

        return Evidence(
            evidence_id=evidence_id,
            kind=EvidenceKind.SPATIAL_CONTEXT,
            value=confidence,
            quality=0.85,
            source="gfw_retrospective",
            timestamp_utc=event_time,
            uncertainty=0.15,
            supports="RETROSPECTIVE_AUDIT",
            references=(
                detection_id,
                "is_retrospective=True",
                "trigger_critical_allowed=false",
                f"mmsi={record_id}",
                f"vessel_name={vessel_name}",
                f"event_type={event_type}",
                f"latency_hours={latency_hours:.2f}",
                "role=retrospective_audit_trail",
            ),
            provenance=prov,
        )

    def audit_detection(
        self,
        gfw_record: Mapping[str, Any],
        detection: Detection,
        alert_priority: Optional[AlertPriority] = None,
    ) -> GfwAuditReport:
        ev = self.generate_evidence(
            gfw_record=gfw_record,
            detection_id=detection.detection_id,
            provenance=detection.provenance,
            alert_priority=alert_priority,
        )
        report_id = f"gfw-rep-{uuid.uuid4().hex[:12]}"
        vessel_name = str(gfw_record.get("vessel_name", "UNKNOWN"))
        mmsi = str(gfw_record.get("mmsi") or "") or None
        lat = gfw_record.get("lat")
        lon = gfw_record.get("lon")

        summary = (
            f"GFW retrospective audit: event {gfw_record.get('event_type', 'EVENT')} "
            f"for vessel {vessel_name} (MMSI: {mmsi}) recorded at ({lat}, {lon}) "
            f"with latency {ev.references[6].split('=')[1]}h. Non-real-time audit context."
        )

        return GfwAuditReport(
            report_id=report_id,
            detection_id=detection.detection_id,
            mmsi=mmsi,
            vessel_name=vessel_name,
            event_timestamp_utc=ev.timestamp_utc,
            latency_hours=float(ev.references[6].split("=")[1]),
            is_retrospective=True,
            evidence=ev,
            provenance=ev.provenance,
            spatial_summary=summary,
            confidence=ev.value,
        )
