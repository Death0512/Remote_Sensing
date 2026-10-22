from datetime import datetime, timezone
import hashlib
from typing import Any, Mapping, Optional, Sequence
import uuid

from src.domain.enums import AlertPriority, Capability, EvidenceKind
from src.domain.evidence import Evidence
from src.domain.provenance import Provenance


class GfwRetrospectiveValidator:
    capability: Capability = Capability.IMPLEMENTED

    def __init__(self, latency_threshold_hours: float = 2.0) -> None:
        self.latency_threshold_hours = latency_threshold_hours

    def _ensure_utc(self, dt: datetime) -> datetime:
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def validate_gfw_event(
        self,
        gfw_record: Mapping[str, Any],
        detection_id: str,
        provenance: Provenance,
        alert_priority: Optional[AlertPriority] = None,
    ) -> Evidence:
        if alert_priority == AlertPriority.CRITICAL:
            raise ValueError("GFW retrospective data cannot trigger real-time critical priority alerts")

        record_id = str(gfw_record.get("mmsi") or gfw_record.get("event_id") or uuid.uuid4())
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

        is_retrospective = True

        evidence_id = f"ev-gfw-{uuid.uuid4()}"
        input_hash = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
        output_hash = hashlib.sha256(evidence_id.encode("utf-8")).hexdigest()

        prov = Provenance(
            provenance_id=str(uuid.uuid4()),
            source_asset_id=provenance.source_asset_id,
            acquisition_metadata_ref=provenance.acquisition_metadata_ref,
            pipeline_version=provenance.pipeline_version,
            git_commit=provenance.git_commit,
            config_hash=f"gfw-retrospective-{self.latency_threshold_hours}",
            model_id=None,
            model_version=None,
            input_checksum=input_hash,
            output_checksum=output_hash,
            processing_run_id=provenance.processing_run_id,
            processing_time_utc=now_utc,
        )

        confidence = float(gfw_record.get("confidence", 0.8))
        vessel_name = str(gfw_record.get("vessel_name", "UNKNOWN"))

        return Evidence(
            evidence_id=evidence_id,
            kind=EvidenceKind.AIS_CORRELATION,
            value=confidence,
            quality=0.85,
            source="gfw_retrospective",
            timestamp_utc=event_time,
            uncertainty=0.15,
            supports="RETROSPECTIVE_AUDIT",
            references=(
                detection_id,
                f"is_retrospective={is_retrospective}",
                f"mmsi={record_id}",
                f"vessel_name={vessel_name}",
                f"latency_hours={latency_hours:.2f}",
                "trigger_critical_allowed=false",
            ),
            provenance=prov,
        )
