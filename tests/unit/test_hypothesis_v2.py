from datetime import datetime, timezone
import pytest

from src.domain.enums import (
    AisCorrelationState,
    CalibrationState,
    Capability,
    EvidenceKind,
    HypothesisStatement,
    ScoreMethod,
    TargetClass,
    Zone,
)
from src.domain.evidence import Evidence
from src.domain.provenance import Provenance
from src.intelligence.accumulator import EvidenceAccumulator
from src.intelligence.hypothesis.engine import HypothesisEngine


def _make_provenance(prov_id: str = "prov-01", asset_id: str = "asset-01") -> Provenance:
    return Provenance(
        provenance_id=prov_id,
        source_asset_id=asset_id,
        acquisition_metadata_ref="s3://imagery/meta.xml",
        pipeline_version="2.0.0",
        git_commit="30d2bc8",
        config_hash="cfg-hyp-test",
        model_id=None,
        model_version=None,
        input_checksum="input-chk-1",
        output_checksum="output-chk-1",
        processing_run_id="run-hyp-test",
        processing_time_utc=datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc),
    )


def _make_maritime_detection_evidence(
    ev_id: str = "ev-maritime-01",
    prov_id: str = "prov-maritime-01",
    source: str = "SarVesselDetector",
) -> Evidence:
    return Evidence(
        evidence_id=ev_id,
        kind=EvidenceKind.OBJECT_CONFIDENCE,
        value=0.92,
        quality=0.95,
        source=source,
        timestamp_utc=datetime(2026, 10, 22, 10, 30, 0, tzinfo=timezone.utc),
        uncertainty=0.08,
        supports=TargetClass.VESSEL.value,
        references=("sensor=S1_SAR", "target_class=VESSEL", "snr_db=15.2"),
        provenance=_make_provenance(prov_id),
    )


def _make_ais_evidence(
    state: AisCorrelationState,
    ev_id: str = "ev-ais-01",
    prov_id: str = "prov-ais-01",
    feed_active: bool = True,
) -> Evidence:
    refs = [
        f"state={state.value}",
        f"feed_active={feed_active}",
        "spatial_gate_m=3000.0",
        "temporal_gate_s=600.0",
    ]
    if feed_active and state == AisCorrelationState.NO_MATCH:
        refs.append("coverage=VERIFIED")

    return Evidence(
        evidence_id=ev_id,
        kind=EvidenceKind.AIS_CORRELATION,
        value=1.0 if state == AisCorrelationState.MATCHED else 0.0,
        quality=1.0 if feed_active else 0.2,
        source="AisCorrelator",
        timestamp_utc=datetime(2026, 10, 22, 10, 30, 0, tzinfo=timezone.utc),
        uncertainty=0.0,
        supports=state.value,
        references=tuple(refs),
        provenance=_make_provenance(prov_id),
    )


def test_dark_vessel_hypothesis_requires_verified_no_match() -> None:
    engine = HypothesisEngine()
    assert engine.capability == Capability.IMPLEMENTED

    acc_valid = EvidenceAccumulator(target_id="target-01")
    maritime_ev = _make_maritime_detection_evidence()
    ais_no_match = _make_ais_evidence(AisCorrelationState.NO_MATCH)
    acc_valid.add(maritime_ev)
    acc_valid.add(ais_no_match)

    hyps_valid = engine.evaluate(acc_valid)
    assert len(hyps_valid) == 1
    hyp_dark = hyps_valid[0]
    assert hyp_dark.statement == HypothesisStatement.DARK_VESSEL_TRANSITING
    assert maritime_ev in hyp_dark.supporting_evidence
    assert ais_no_match in hyp_dark.supporting_evidence
    assert hyp_dark.score_method == ScoreMethod.RULE_ENGINE
    assert hyp_dark.calibration_state == CalibrationState.NOT_CALIBRATED

    acc_unavail = EvidenceAccumulator(target_id="target-02")
    ais_unavail = _make_ais_evidence(
        AisCorrelationState.AIS_UNAVAILABLE,
        ev_id="ev-ais-unavail",
        prov_id="prov-ais-unavail",
        feed_active=False,
    )
    acc_unavail.add(maritime_ev)
    acc_unavail.add(ais_unavail)

    hyps_unavail = engine.evaluate(acc_unavail)
    assert len(hyps_unavail) == 1
    hyp_rejected = hyps_unavail[0]
    assert hyp_rejected.statement != HypothesisStatement.DARK_VESSEL_TRANSITING
    assert hyp_rejected.statement != HypothesisStatement.DARK_VESSEL_CANDIDATE
    assert hyp_rejected.statement == HypothesisStatement.UNVERIFIED_SURFACE_TARGET


def test_unverified_surface_target_when_ais_unavailable() -> None:
    engine = HypothesisEngine()
    acc = EvidenceAccumulator(target_id="target-03")
    maritime_ev = _make_maritime_detection_evidence()
    ais_stale = _make_ais_evidence(
        AisCorrelationState.AIS_STALE,
        ev_id="ev-ais-stale",
        prov_id="prov-ais-stale",
        feed_active=True,
    )
    acc.add(maritime_ev)
    acc.add(ais_stale)

    hyps = engine.evaluate(acc)
    assert len(hyps) == 1
    hyp = hyps[0]
    assert hyp.statement == HypothesisStatement.UNVERIFIED_SURFACE_TARGET
    assert "telemetry" in hyp.primary_explanation.lower() or "ais" in hyp.primary_explanation.lower()
    assert "stale" in hyp.detailed_rationale.lower() or "unavailable" in hyp.detailed_rationale.lower()


def test_air_incursion_hypothesis_from_kinematics() -> None:
    engine = HypothesisEngine()
    acc = EvidenceAccumulator(target_id="air-01")

    opt_ev = Evidence(
        evidence_id="ev-opt-01",
        kind=EvidenceKind.OBJECT_CONFIDENCE,
        value=0.88,
        quality=0.85,
        source="OpticalAircraftDetector",
        timestamp_utc=datetime(2026, 10, 22, 11, 0, 0, tzinfo=timezone.utc),
        uncertainty=0.12,
        supports=TargetClass.AIRCRAFT.value,
        references=("sensor=VHR_OPTICAL", "target_class=AIRCRAFT"),
        provenance=_make_provenance("prov-air-det"),
    )

    kin_ev = Evidence(
        evidence_id="ev-kin-01",
        kind=EvidenceKind.KINEMATICS,
        value=650.0,
        quality=0.9,
        source="ParallaxInflightDetector",
        timestamp_utc=datetime(2026, 10, 22, 11, 0, 0, tzinfo=timezone.utc),
        uncertainty=25.0,
        supports="EXPERIMENTAL_INFLIGHT_TARGET",
        references=("speed_kmh=650.0",),
        provenance=_make_provenance("prov-air-kin"),
    )

    zone_ev = Evidence(
        evidence_id="ev-zone-01",
        kind=EvidenceKind.SPATIAL_CONTEXT,
        value=1.0,
        quality=1.0,
        source="GeofenceEngine",
        timestamp_utc=datetime(2026, 10, 22, 11, 0, 0, tzinfo=timezone.utc),
        uncertainty=0.0,
        supports=Zone.ZONE_1.value,
        references=("zone=ZONE_1",),
        provenance=_make_provenance("prov-air-zone"),
    )

    acc.add(opt_ev)
    acc.add(kin_ev)
    acc.add(zone_ev)

    hyps = engine.evaluate(acc)
    assert len(hyps) == 1
    hyp = hyps[0]
    assert hyp.statement == HypothesisStatement.UNIDENTIFIED_AIR_INCURSION
    assert opt_ev in hyp.supporting_evidence
    assert kin_ev in hyp.supporting_evidence
    assert "territorial airspace" in hyp.primary_explanation.lower() or "airborne" in hyp.primary_explanation.lower()


def test_hypothesis_explanation_formatting() -> None:
    engine = HypothesisEngine()
    acc = EvidenceAccumulator()
    acc.add(_make_maritime_detection_evidence())
    acc.add(_make_ais_evidence(AisCorrelationState.NO_MATCH))

    hyps = engine.evaluate(acc)
    assert len(hyps) == 1
    hyp = hyps[0]

    assert hyp.primary_explanation
    assert len(hyp.primary_explanation.strip()) > 0
    assert hyp.detailed_rationale
    assert len(hyp.detailed_rationale.strip()) > 0

    assert "Calibrated probability unavailable; heuristic/model raw score:" in hyp.detailed_rationale
    assert hyp.score is not None
    assert 0.0 <= hyp.score <= 1.0
    assert len(hyp.evidence_ids) >= 2


def test_dark_vessel_conjunction() -> None:
    engine = HypothesisEngine()

    maritime_ev = _make_maritime_detection_evidence()
    matched_ais = _make_ais_evidence(AisCorrelationState.MATCHED)
    acc_matched = EvidenceAccumulator()
    acc_matched.add(maritime_ev)
    acc_matched.add(matched_ais)

    hyps_matched = engine.evaluate(acc_matched)
    assert len(hyps_matched) == 1
    assert hyps_matched[0].statement != HypothesisStatement.DARK_VESSEL_TRANSITING

    acc_no_ais = EvidenceAccumulator()
    acc_no_ais.add(maritime_ev)
    hyps_no_ais = engine.evaluate(acc_no_ais)
    assert len(hyps_no_ais) == 1
    assert hyps_no_ais[0].statement != HypothesisStatement.DARK_VESSEL_TRANSITING
    assert hyps_no_ais[0].statement == HypothesisStatement.UNVERIFIED_SURFACE_TARGET

    acc_inactive_feed = EvidenceAccumulator()
    inactive_ais = _make_ais_evidence(
        AisCorrelationState.NO_MATCH,
        feed_active=False,
    )
    acc_inactive_feed.add(maritime_ev)
    acc_inactive_feed.add(inactive_ais)
    hyps_inactive = engine.evaluate(acc_inactive_feed)
    assert len(hyps_inactive) == 1
    assert hyps_inactive[0].statement != HypothesisStatement.DARK_VESSEL_TRANSITING
    assert hyps_inactive[0].statement == HypothesisStatement.UNVERIFIED_SURFACE_TARGET


def test_coastal_change_hypothesis() -> None:
    engine = HypothesisEngine()
    acc = EvidenceAccumulator()

    chg_ev = Evidence(
        evidence_id="ev-chg-01",
        kind=EvidenceKind.CHANGE_STRENGTH,
        value=0.78,
        quality=0.92,
        source="BitemporalChangeDetector",
        timestamp_utc=datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc),
        uncertainty=0.08,
        supports="INFRASTRUCTURE_EXPANSION_DETECTED",
        references=("residual_px=0.32", "quality_tier=HIGH", "method=log_ratio"),
        provenance=_make_provenance("prov-chg"),
    )

    surface_corrob = Evidence(
        evidence_id="ev-corrob-01",
        kind=EvidenceKind.OBJECT_CONFIDENCE,
        value=0.85,
        quality=0.88,
        source="OpticalSurfaceValidator",
        timestamp_utc=datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc),
        uncertainty=0.10,
        supports="SURFACE_CONFIRMATION",
        references=("sensor=S2_MSI",),
        provenance=_make_provenance("prov-corrob"),
    )

    acc.add(chg_ev)
    acc.add(surface_corrob)

    hyps = engine.evaluate(acc)
    assert len(hyps) == 1
    hyp = hyps[0]
    assert hyp.statement == HypothesisStatement.INFRASTRUCTURE_EXPANSION_DETECTED
    assert chg_ev in hyp.supporting_evidence
    assert surface_corrob in hyp.supporting_evidence
    assert "change" in hyp.primary_explanation.lower() or "infrastructure" in hyp.primary_explanation.lower()


def test_accumulator_deduplication_and_query() -> None:
    acc = EvidenceAccumulator(target_id="target-dedup")

    ev1 = _make_maritime_detection_evidence(ev_id="ev-1", prov_id="prov-same")
    ev2 = _make_maritime_detection_evidence(ev_id="ev-2", prov_id="prov-same")
    ev3 = _make_maritime_detection_evidence(ev_id="ev-3", prov_id="prov-diff")

    added1 = acc.add(ev1)
    assert added1 is True
    added2 = acc.add(ev2)
    assert added2 is False
    added3 = acc.add(ev3)
    assert added3 is True

    assert len(acc) == 2
    assert "ev-1" in acc
    assert "ev-2" not in acc
    assert "ev-3" in acc

    queried_object = acc.get_by_kind(EvidenceKind.OBJECT_CONFIDENCE)
    assert len(queried_object) == 2
    queried_ais = acc.get_by_kind(EvidenceKind.AIS_CORRELATION)
    assert len(queried_ais) == 0

    acc.clear()
    assert len(acc) == 0
