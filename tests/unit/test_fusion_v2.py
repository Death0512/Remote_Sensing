from datetime import datetime, timedelta, timezone
import pytest
from shapely.geometry import Point

from src.domain.confidence import ConfidenceVector
from src.domain.detection import Detection
from src.domain.enums import (
    AisCorrelationState,
    AlertPriority,
    Capability,
    EvidenceKind,
    SensorType,
    TargetClass,
)
from src.domain.provenance import Provenance
from src.fusion.ais.correlator import AisCorrelator, AisRecord
from src.fusion.ais.gfw_retrospective import GfwRetrospectiveValidator


def _create_provenance(prov_id: str = "prov-001") -> Provenance:
    return Provenance(
        provenance_id=prov_id,
        source_asset_id="asset-test",
        acquisition_metadata_ref="s3://test/meta.xml",
        pipeline_version="2.0.0",
        git_commit="fedcba9",
        config_hash="cfg-fusion-01",
        model_id=None,
        model_version=None,
        input_checksum="input-chk",
        output_checksum="output-chk",
        processing_run_id="run-fusion-test",
        processing_time_utc=datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc),
    )


def _create_detection(
    det_id: str,
    lon: float,
    lat: float,
    acq_time: datetime,
    speed_knots: float = 12.0,
    heading_deg: float = 90.0,
) -> Detection:
    return Detection(
        detection_id=det_id,
        candidate_id=f"cand-{det_id}",
        sensor=SensorType.S1_SAR,
        geom_wgs84=Point(lon, lat),
        geom_px=(100.0, 100.0, 120.0, 110.0),
        crs="EPSG:4326",
        target_class=TargetClass.VESSEL,
        confidence=ConfidenceVector(raw_score=0.92),
        attributes=(("speed_knots", speed_knots), ("heading_deg", heading_deg)),
        acquisition_time_utc=acq_time,
        provenance=_create_provenance(f"prov-{det_id}"),
    )


def test_ais_down_yields_unavailable_not_dark() -> None:
    correlator = AisCorrelator()
    assert correlator.capability == Capability.IMPLEMENTED

    t0 = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)
    det = _create_detection("det-1", 108.0, 16.0, t0)

    evidence = correlator.correlate(
        detection=det,
        ais_records=[],
        feed_active=False,
        last_feed_time_utc=t0,
    )

    assert evidence.kind == EvidenceKind.AIS_CORRELATION
    assert evidence.supports == AisCorrelationState.AIS_UNAVAILABLE.value
    assert "state=AIS_UNAVAILABLE" in evidence.references
    assert "DARK_VESSEL" not in evidence.supports
    assert "DARK_VESSEL" not in "".join(evidence.references)


def test_stale_ais_feed_yields_stale() -> None:
    correlator = AisCorrelator(max_stale_seconds=1800.0)
    t0 = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)
    det = _create_detection("det-2", 108.0, 16.0, t0)

    stale_feed_time = t0 - timedelta(seconds=2000)

    evidence = correlator.correlate(
        detection=det,
        ais_records=[],
        feed_active=True,
        last_feed_time_utc=stale_feed_time,
    )

    assert evidence.kind == EvidenceKind.AIS_CORRELATION
    assert evidence.supports == AisCorrelationState.AIS_STALE.value
    assert "state=AIS_STALE" in evidence.references


def test_matched_ais_record() -> None:
    correlator = AisCorrelator()
    t0 = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)
    det = _create_detection("det-3", 108.0, 16.0, t0, speed_knots=10.0, heading_deg=90.0)

    record = AisRecord(
        mmsi="123456789",
        lat=16.002,
        lon=108.002,
        timestamp=t0 + timedelta(seconds=30),
        vessel_name="PACIFIC_TRADER",
        speed_knots=10.5,
        heading_deg=92.0,
    )

    evidence = correlator.correlate(
        detection=det,
        ais_records=[record],
        feed_active=True,
        last_feed_time_utc=t0,
    )

    assert evidence.kind == EvidenceKind.AIS_CORRELATION
    assert evidence.supports == AisCorrelationState.MATCHED.value
    assert "state=MATCHED" in evidence.references
    assert "mmsi=123456789" in evidence.references


def test_partial_match_on_kinematics() -> None:
    correlator = AisCorrelator(speed_tolerance_knots=5.0, heading_tolerance_deg=30.0)
    t0 = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)
    det = _create_detection("det-3b", 108.0, 16.0, t0, speed_knots=20.0, heading_deg=0.0)

    record = AisRecord(
        mmsi="987654321",
        lat=16.001,
        lon=108.001,
        timestamp=t0 + timedelta(seconds=10),
        vessel_name="SPEEDY",
        speed_knots=5.0,
        heading_deg=180.0,
    )

    evidence = correlator.correlate(
        detection=det,
        ais_records=[record],
        feed_active=True,
        last_feed_time_utc=t0,
    )

    assert evidence.kind == EvidenceKind.AIS_CORRELATION
    assert evidence.supports == AisCorrelationState.PARTIAL_MATCH.value
    assert "state=PARTIAL_MATCH" in evidence.references


def test_active_feed_without_match_yields_no_match() -> None:
    correlator = AisCorrelator()
    t0 = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)
    det = _create_detection("det-4", 108.0, 16.0, t0)

    far_record = AisRecord(
        mmsi="999888777",
        lat=16.5,
        lon=108.5,
        timestamp=t0,
        vessel_name="DISTANT_SHIP",
    )

    evidence = correlator.correlate(
        detection=det,
        ais_records=[far_record],
        feed_active=True,
        last_feed_time_utc=t0,
        spatial_gate_m=3000.0,
    )

    assert evidence.kind == EvidenceKind.AIS_CORRELATION
    assert evidence.supports == AisCorrelationState.NO_MATCH.value
    assert "state=NO_MATCH" in evidence.references
    assert "DARK_VESSEL" not in evidence.supports


def test_ambiguous_multiple_matches_yields_uncertain() -> None:
    correlator = AisCorrelator()
    t0 = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)
    det = _create_detection("det-5", 108.0, 16.0, t0)

    rec1 = AisRecord(
        mmsi="111111111",
        lat=16.001,
        lon=108.001,
        timestamp=t0,
    )
    rec2 = AisRecord(
        mmsi="222222222",
        lat=16.0012,
        lon=108.0012,
        timestamp=t0,
    )

    evidence = correlator.correlate(
        detection=det,
        ais_records=[rec1, rec2],
        feed_active=True,
        last_feed_time_utc=t0,
        spatial_gate_m=3000.0,
    )

    assert evidence.kind == EvidenceKind.AIS_CORRELATION
    assert evidence.supports == AisCorrelationState.CORRELATION_UNCERTAIN.value
    assert "state=CORRELATION_UNCERTAIN" in evidence.references
    assert any("candidate_mmsi=111111111" in r for r in evidence.references)
    assert any("candidate_mmsi=222222222" in r for r in evidence.references)


def test_gfw_retrospective_evidence_contract() -> None:
    validator = GfwRetrospectiveValidator()
    assert validator.capability == Capability.IMPLEMENTED

    prov = _create_provenance("prov-gfw")
    gfw_data = {
        "mmsi": "412000001",
        "vessel_name": "SEA_EXPLORER",
        "timestamp": "2026-10-22T02:00:00Z",
        "confidence": 0.95,
    }

    evidence = validator.validate_gfw_event(
        gfw_record=gfw_data,
        detection_id="det-gfw-1",
        provenance=prov,
    )

    assert evidence.kind == EvidenceKind.AIS_CORRELATION
    assert evidence.supports == "RETROSPECTIVE_AUDIT"
    assert "is_retrospective=True" in evidence.references
    assert "trigger_critical_allowed=false" in evidence.references
    assert "mmsi=412000001" in evidence.references

    with pytest.raises(ValueError, match="GFW retrospective data cannot trigger real-time critical priority alerts"):
        validator.validate_gfw_event(
            gfw_record=gfw_data,
            detection_id="det-gfw-1",
            provenance=prov,
            alert_priority=AlertPriority.CRITICAL,
        )
