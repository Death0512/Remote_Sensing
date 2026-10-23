from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import math
from typing import Optional
import pytest
from shapely.geometry import Point, Polygon

from src.cli.main import cli_entrypoint
from src.domain.association import Association
from src.domain.confidence import ConfidenceVector
from src.domain.detection import Detection
from src.domain.enums import (
    AisCorrelationState,
    AlertPriority,
    Capability,
    EvidenceKind,
    HypothesisStatement,
    SensorType,
    TargetClass,
    Zone,
)
from src.domain.evidence import Evidence
from src.domain.provenance import Provenance
from src.intelligence.ais.correlator import (
    AisCorrelator,
    AisCoverageModel,
    AisCoverageZone,
    AisRecord,
)
from src.intelligence.gfw.audit import GfwAuditReport, GfwRetrospectiveAuditor
from src.intelligence.hypothesis.engine import HypothesisEngine
from tests.mocks.mock_data import generate_synthetic_sar_scene


def _make_provenance(prov_id: str = "prov-001") -> Provenance:
    return Provenance(
        provenance_id=prov_id,
        source_asset_id="asset-test-ais",
        acquisition_metadata_ref="s3://test/meta.xml",
        pipeline_version="3.0.0",
        git_commit="fedcba9",
        config_hash="cfg-ais-test",
        model_id=None,
        model_version=None,
        input_checksum="input-chk-ais",
        output_checksum="output-chk-ais",
        processing_run_id="run-ais-test",
        processing_time_utc=datetime(2026, 10, 23, 10, 0, 0, tzinfo=timezone.utc),
    )


def _make_detection(
    det_id: str = "det-test-01",
    lon: float = 108.0,
    lat: float = 16.0,
    acq_time: Optional[datetime] = None,
    speed_knots: Optional[float] = 12.0,
    heading_deg: Optional[float] = 90.0,
    uncertainty_m: float = 30.0,
) -> Detection:
    t = acq_time or datetime(2026, 10, 23, 10, 0, 0, tzinfo=timezone.utc)
    attrs = []
    if speed_knots is not None:
        attrs.append(("speed_knots", speed_knots))
    if heading_deg is not None:
        attrs.append(("heading_deg", heading_deg))
    attrs.append(("uncertainty_m", uncertainty_m))

    return Detection(
        detection_id=det_id,
        candidate_id=f"cand-{det_id}",
        sensor=SensorType.S1_SAR,
        geom_wgs84=Point(lon, lat),
        geom_px=(100.0, 100.0, 120.0, 110.0),
        crs="EPSG:4326",
        target_class=TargetClass.VESSEL,
        confidence=ConfidenceVector(raw_score=0.91),
        attributes=tuple(attrs),
        acquisition_time_utc=t,
        provenance=_make_provenance(f"prov-{det_id}"),
    )


def _make_coverage_model(
    center_lon: float = 108.0,
    center_lat: float = 16.0,
    delta: float = 0.5,
) -> AisCoverageModel:
    poly = Polygon(
        [
            (center_lon - delta, center_lat - delta),
            (center_lon + delta, center_lat - delta),
            (center_lon + delta, center_lat + delta),
            (center_lon - delta, center_lat + delta),
        ]
    )
    zone = AisCoverageZone(zone_id="coastal-coverage-1", geometry=poly, active=True)
    return AisCoverageModel(zones=(zone,))


def test_ais_states_matched():
    t0 = datetime(2026, 10, 23, 10, 0, 0, tzinfo=timezone.utc)
    cov = _make_coverage_model()
    correlator = AisCorrelator(coverage_model=cov)
    det = _make_detection("det-matched", 108.0, 16.0, t0, speed_knots=12.0, heading_deg=90.0)

    rec = AisRecord(
        mmsi="111222333",
        lat=16.0,
        lon=108.001,
        timestamp=t0 + timedelta(seconds=10),
        vessel_name="PACIFIC_NAVIGATOR",
        speed_knots=12.2,
        heading_deg=89.5,
    )

    assoc = correlator.associate(det, [rec], feed_active=True, last_feed_time_utc=t0)
    assert isinstance(assoc, Association)
    assert assoc.association_type == AisCorrelationState.MATCHED.value
    assert assoc.target_id == "111222333"
    assert assoc.residual < 500.0
    assert assoc.confidence > 0.7
    assert "KINEMATIC_VERIFIED" in assoc.quality_flags
    assert "COVERAGE_VERIFIED" in assoc.quality_flags

    ev = correlator.to_evidence(assoc)
    assert ev.kind == EvidenceKind.AIS_CORRELATION
    assert ev.supports == AisCorrelationState.MATCHED.value


def test_ais_states_partial_match_on_kinematic_mismatch():
    t0 = datetime(2026, 10, 23, 10, 0, 0, tzinfo=timezone.utc)
    cov = _make_coverage_model()
    correlator = AisCorrelator(
        coverage_model=cov,
        speed_tolerance_knots=4.0,
        heading_tolerance_deg=30.0,
    )
    det = _make_detection("det-partial", 108.0, 16.0, t0, speed_knots=10.0, heading_deg=0.0)

    rec = AisRecord(
        mmsi="222333444",
        lat=16.0005,
        lon=108.0005,
        timestamp=t0,
        speed_knots=25.0,
        heading_deg=180.0,
    )

    assoc = correlator.associate(det, [rec], feed_active=True, last_feed_time_utc=t0)
    assert assoc.association_type == AisCorrelationState.PARTIAL_MATCH.value
    assert assoc.target_id == "222333444"
    assert "KINEMATIC_MISMATCH" in assoc.quality_flags


def test_coverage_required_for_no_match():
    t0 = datetime(2026, 10, 23, 10, 0, 0, tzinfo=timezone.utc)
    cov = _make_coverage_model(center_lon=108.0, center_lat=16.0, delta=0.2)
    correlator = AisCorrelator(coverage_model=cov)

    det_outside = _make_detection("det-out", 112.0, 20.0, t0)
    assoc_out = correlator.associate(det_outside, [], feed_active=True, last_feed_time_utc=t0)
    assert assoc_out.association_type == AisCorrelationState.AIS_UNAVAILABLE.value
    assert "OUTSIDE_COVERAGE" in assoc_out.quality_flags
    assert assoc_out.association_type != AisCorrelationState.NO_MATCH.value

    det_inside = _make_detection("det-in", 108.0, 16.0, t0)
    assoc_in = correlator.associate(det_inside, [], feed_active=True, last_feed_time_utc=t0)
    assert assoc_in.association_type == AisCorrelationState.NO_MATCH.value
    assert "COVERAGE_VERIFIED" in assoc_in.quality_flags

    correlator_no_cov = AisCorrelator(coverage_model=None)
    assoc_no_cov = correlator_no_cov.associate(det_inside, [], feed_active=True, last_feed_time_utc=t0)
    assert assoc_no_cov.association_type == AisCorrelationState.AIS_UNAVAILABLE.value
    assert "NO_COVERAGE_MODEL" in assoc_no_cov.quality_flags


def test_ais_down_unavailable():
    t0 = datetime(2026, 10, 23, 10, 0, 0, tzinfo=timezone.utc)
    cov = _make_coverage_model()
    correlator = AisCorrelator(coverage_model=cov)
    det = _make_detection("det-down", 108.0, 16.0, t0)

    assoc = correlator.associate(det, [], feed_active=False, last_feed_time_utc=t0)
    assert assoc.association_type == AisCorrelationState.AIS_UNAVAILABLE.value
    assert "FEED_INACTIVE" in assoc.quality_flags
    assert assoc.target_id == "NONE"

    ev = correlator.to_evidence(assoc)
    engine = HypothesisEngine()
    maritime_ev = Evidence(
        evidence_id="ev-sar-down",
        kind=EvidenceKind.OBJECT_CONFIDENCE,
        value=0.92,
        quality=1.0,
        source="SarVesselDetector",
        timestamp_utc=t0,
        uncertainty=0.08,
        supports=TargetClass.VESSEL.value,
        references=("sensor=S1_SAR", "target_class=VESSEL"),
        provenance=det.provenance,
    )
    hyps = engine.evaluate([maritime_ev, ev])
    assert len(hyps) == 1
    assert hyps[0].statement == HypothesisStatement.UNVERIFIED_SURFACE_TARGET
    assert hyps[0].statement != HypothesisStatement.DARK_VESSEL_TRANSITING
    assert hyps[0].statement != HypothesisStatement.DARK_VESSEL_CANDIDATE


def test_ais_states_stale():
    t0 = datetime(2026, 10, 23, 10, 0, 0, tzinfo=timezone.utc)
    cov = _make_coverage_model()
    correlator = AisCorrelator(coverage_model=cov, max_stale_seconds=1800.0)
    det = _make_detection("det-stale", 108.0, 16.0, t0)

    stale_time = t0 - timedelta(seconds=2500)
    assoc = correlator.associate(det, [], feed_active=True, last_feed_time_utc=stale_time)
    assert assoc.association_type == AisCorrelationState.AIS_STALE.value
    assert any("FEED_STALE" in f for f in assoc.quality_flags)


def test_ais_states_correlation_uncertain_on_multi_candidate():
    t0 = datetime(2026, 10, 23, 10, 0, 0, tzinfo=timezone.utc)
    cov = _make_coverage_model()
    correlator = AisCorrelator(coverage_model=cov)
    det = _make_detection("det-uncertain", 108.0, 16.0, t0)

    rec1 = AisRecord(mmsi="111111111", lat=16.001, lon=108.001, timestamp=t0)
    rec2 = AisRecord(mmsi="222222222", lat=16.0012, lon=108.0011, timestamp=t0)

    assoc = correlator.associate(det, [rec1, rec2], feed_active=True, last_feed_time_utc=t0)
    assert assoc.association_type == AisCorrelationState.CORRELATION_UNCERTAIN.value
    assert any("AMBIGUOUS_MULTI_CANDIDATE" in f for f in assoc.quality_flags)


def test_uncertainty_propagation_kinematics():
    correlator = AisCorrelator(default_max_speed_mps=15.0)
    sigma_meas = 30.0
    sigma_ais = 20.0
    dt = 60.0
    v_max = 15.0

    sigma_pos = correlator.propagate_uncertainty(sigma_meas, sigma_ais, dt, v_max)
    expected = math.sqrt(30.0**2 + 20.0**2 + (15.0 * 60.0) ** 2)
    assert math.isclose(sigma_pos, expected, rel_tol=1e-5)


def test_no_match_not_dark():
    t0 = datetime(2026, 10, 23, 10, 0, 0, tzinfo=timezone.utc)
    cov = _make_coverage_model(center_lon=108.0, center_lat=16.0, delta=0.5)
    correlator = AisCorrelator(coverage_model=cov)
    det = _make_detection("det-nomatch", 108.0, 16.0, t0)

    far_rec = AisRecord(mmsi="999999999", lat=17.5, lon=109.5, timestamp=t0)
    assoc = correlator.associate(det, [far_rec], feed_active=True, last_feed_time_utc=t0)
    assert assoc.association_type == AisCorrelationState.NO_MATCH.value

    engine = HypothesisEngine()
    maritime_ev = Evidence(
        evidence_id="ev-sar-nomatch",
        kind=EvidenceKind.OBJECT_CONFIDENCE,
        value=0.92,
        quality=1.0,
        source="SarVesselDetector",
        timestamp_utc=t0,
        uncertainty=0.08,
        supports=TargetClass.VESSEL.value,
        references=("sensor=S1_SAR", "target_class=VESSEL"),
        provenance=det.provenance,
    )
    ais_ev = correlator.to_evidence(assoc)
    hyps = engine.evaluate([maritime_ev, ais_ev])
    assert len(hyps) == 1
    assert hyps[0].statement == HypothesisStatement.DARK_VESSEL_TRANSITING


def test_associate_batch_one_to_one_deduplication():
    t0 = datetime(2026, 10, 23, 10, 0, 0, tzinfo=timezone.utc)
    cov = _make_coverage_model()
    correlator = AisCorrelator(coverage_model=cov)

    det1 = _make_detection("det-b1", 108.0, 16.0, t0)
    det2 = _make_detection("det-b2", 108.001, 16.001, t0)

    rec_shared = AisRecord(mmsi="555666777", lat=16.0001, lon=108.0001, timestamp=t0)
    rec_second = AisRecord(mmsi="888999000", lat=16.0011, lon=108.0011, timestamp=t0)

    assocs = correlator.associate_batch(
        detections=[det1, det2],
        ais_records=[rec_shared, rec_second],
        feed_active=True,
        last_feed_time_utc=t0,
    )

    assert len(assocs) == 2
    assigned_targets = [a.target_id for a in assocs if a.target_id != "NONE"]
    assert len(assigned_targets) == len(set(assigned_targets))


def test_elimination_of_zone_to_dark_vessel_shortcut(tmp_path: Path):
    sar_scene = generate_synthetic_sar_scene(
        height=100,
        width=100,
        targets=[(30, 30, 4, 4, 300.0)],
        seed=42,
    )
    img_path = tmp_path / "zone1_scene.npy"
    import numpy as np
    np.save(str(img_path), sar_scene)

    out_dir_no_ais = tmp_path / "out_no_ais"
    ret = cli_entrypoint([
        "scan",
        "--image", str(img_path),
        "--sensor", "S1_GRD",
        "--output-dir", str(out_dir_no_ais),
    ])
    assert ret == 0

    alerts_file = out_dir_no_ais / "alerts.geojson"
    assert alerts_file.exists()
    with open(alerts_file) as f:
        data = json.load(f)

    assert len(data["features"]) > 0
    feat = data["features"][0]
    assert feat["properties"]["zone"] == 1
    assert feat["properties"]["is_dark"] is False
    assert feat["properties"]["hypothesis"]["statement"] == HypothesisStatement.UNVERIFIED_SURFACE_TARGET.value
    assert feat["properties"]["hypothesis"]["statement"] != HypothesisStatement.DARK_VESSEL_TRANSITING.value


def test_conjunction_rule_enables_dark_vessel_in_cli(tmp_path: Path):
    sar_scene = generate_synthetic_sar_scene(
        height=100,
        width=100,
        targets=[(30, 30, 4, 4, 300.0)],
        seed=42,
    )
    img_path = tmp_path / "dark_scene.npy"
    import numpy as np
    np.save(str(img_path), sar_scene)

    out_dir_verified = tmp_path / "out_verified"
    ret = cli_entrypoint([
        "scan",
        "--image", str(img_path),
        "--sensor", "S1_GRD",
        "--output-dir", str(out_dir_verified),
        "--verified-ais-coverage",
        "--is-dark",
    ])
    assert ret == 0

    alerts_file = out_dir_verified / "alerts.geojson"
    assert alerts_file.exists()
    with open(alerts_file) as f:
        data = json.load(f)

    assert len(data["features"]) > 0
    feat = data["features"][0]
    assert feat["properties"]["is_dark"] is True
    assert feat["properties"]["hypothesis"]["statement"] == HypothesisStatement.DARK_VESSEL_TRANSITING.value


def test_gfw_retrospective_audit_behavior():
    auditor = GfwRetrospectiveAuditor(latency_threshold_hours=2.0)
    assert auditor.capability == Capability.IMPLEMENTED

    t_past = datetime.now(timezone.utc) - timedelta(hours=5)
    gfw_data = {
        "mmsi": "413123456",
        "vessel_name": "HAI_YANG_68",
        "event_type": "FISHING_LOITERING",
        "timestamp": t_past.isoformat(),
        "lat": 16.2,
        "lon": 108.4,
        "confidence": 0.88,
    }

    prov = _make_provenance("prov-gfw-audit")
    ev = auditor.generate_evidence(
        gfw_record=gfw_data,
        detection_id="det-gfw-test",
        provenance=prov,
    )

    assert ev.kind == EvidenceKind.SPATIAL_CONTEXT
    assert ev.supports == "RETROSPECTIVE_AUDIT"
    assert ev.source == "gfw_retrospective"
    assert "is_retrospective=True" in ev.references
    assert "trigger_critical_allowed=false" in ev.references
    assert "mmsi=413123456" in ev.references

    det = _make_detection("det-gfw-report", 108.4, 16.2, t_past)
    rep = auditor.audit_detection(gfw_data, det)
    assert isinstance(rep, GfwAuditReport)
    assert rep.detection_id == "det-gfw-report"
    assert rep.mmsi == "413123456"
    assert rep.latency_hours >= 4.9
    assert rep.is_retrospective is True

    with pytest.raises(ValueError, match="GFW retrospective data cannot trigger real-time critical priority alerts"):
        auditor.generate_evidence(
            gfw_record=gfw_data,
            detection_id="det-crit",
            provenance=prov,
            alert_priority=AlertPriority.CRITICAL,
        )
