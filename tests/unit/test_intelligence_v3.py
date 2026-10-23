from datetime import datetime, timezone
import pytest

from src.domain.enums import (
    AisCorrelationState,
    CalibrationState,
    EvidenceKind,
    HypothesisStatement,
    ScoreMethod,
    TargetClass,
    Zone,
)
from src.domain.evidence import Evidence
from src.domain.hypothesis import Hypothesis
from src.domain.provenance import Provenance
from src.intelligence.accumulator import EvidenceAccumulator
from src.intelligence.calibration.calibrator import (
    IsotonicCalibrator,
    PlattCalibrator,
    ProbabilityCalibrator,
    ReliabilityCurveCalibrator,
)
from src.intelligence.hypothesis.engine import HypothesisEngine


def _make_provenance(prov_id: str = "prov-001", asset_id: str = "asset-001") -> Provenance:
    return Provenance(
        provenance_id=prov_id,
        source_asset_id=asset_id,
        acquisition_metadata_ref="s3://test/meta.xml",
        pipeline_version="3.0.0",
        git_commit="fedcba9",
        config_hash="cfg-intel-test",
        model_id=None,
        model_version=None,
        input_checksum="input-chk",
        output_checksum="output-chk",
        processing_run_id="run-intel-test",
        processing_time_utc=datetime(2026, 10, 23, 10, 0, 0, tzinfo=timezone.utc),
    )


def _make_evidence(
    ev_id: str = "ev-001",
    kind: EvidenceKind = EvidenceKind.OBJECT_CONFIDENCE,
    value: float = 0.9,
    quality: float = 0.95,
    source: str = "SarVesselDetector",
    supports: str = TargetClass.VESSEL.value,
    references: tuple[str, ...] = ("sensor=S1_SAR", "target_class=VESSEL"),
    prov_id: str = "prov-001",
) -> Evidence:
    return Evidence(
        evidence_id=ev_id,
        kind=kind,
        value=value,
        quality=quality,
        source=source,
        timestamp_utc=datetime(2026, 10, 23, 10, 0, 0, tzinfo=timezone.utc),
        uncertainty=0.05,
        supports=supports,
        references=references,
        provenance=_make_provenance(prov_id),
    )


def test_evidence_accumulator_deduplication_and_provenance():
    acc = EvidenceAccumulator(target_id="tgt-1", track_id="trk-1")
    ev1 = _make_evidence("ev-1", prov_id="prov-1", references=("source_id=src-1",))
    assert acc.add(ev1) is True
    assert len(acc) == 1

    assert acc.add(ev1) is False
    assert len(acc) == 1

    ev2_dup_prov = _make_evidence("ev-2", prov_id="prov-1", references=("source_id=src-2",))
    assert acc.add(ev2_dup_prov) is False
    assert len(acc) == 1

    ev3_dup_src = _make_evidence("ev-3", prov_id="prov-3", references=("source_id=src-1",))
    assert acc.add(ev3_dup_src) is False
    assert len(acc) == 1

    ev4_valid = _make_evidence("ev-4", prov_id="prov-4", references=("source_id=src-4",))
    assert acc.add(ev4_valid) is True
    assert len(acc) == 2


def test_evidence_accumulator_query_and_summary():
    acc = EvidenceAccumulator(target_id="tgt-100", track_id="trk-100")
    ev_sar = _make_evidence(
        "ev-sar",
        kind=EvidenceKind.OBJECT_CONFIDENCE,
        source="SarDetector",
        references=("target_id=tgt-100", "sensor=S1_SAR"),
        prov_id="p-1",
    )
    ev_ais = _make_evidence(
        "ev-ais",
        kind=EvidenceKind.AIS_CORRELATION,
        source="AisCorrelator",
        references=("target_id=tgt-100", "coverage=VERIFIED"),
        prov_id="p-2",
    )
    ev_kin = _make_evidence(
        "ev-kin",
        kind=EvidenceKind.KINEMATICS,
        source="KalmanTracker",
        references=("track_id=trk-100",),
        prov_id="p-3",
    )

    acc.extend([ev_sar, ev_ais, ev_kin])
    assert len(acc.all_evidence()) == 3

    by_kind = acc.get_by_kind(EvidenceKind.AIS_CORRELATION)
    assert len(by_kind) == 1
    assert by_kind[0].evidence_id == "ev-ais"

    queried_source = acc.query(source="SarDetector")
    assert len(queried_source) == 1
    assert queried_source[0].evidence_id == "ev-sar"

    queried_target = acc.query(target_id="tgt-100")
    assert len(queried_target) == 2

    queried_track = acc.query(track_id="trk-100")
    assert len(queried_track) == 1

    queried_custom = acc.query(predicate=lambda e: e.value > 0.5)
    assert len(queried_custom) == 3

    summary = acc.summary()
    assert summary["target_id"] == "tgt-100"
    assert summary["track_id"] == "trk-100"
    assert summary["total_items"] == 3
    assert "OBJECT_CONFIDENCE" in summary["kinds"]


def test_calibration_platt():
    cal = PlattCalibrator()
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    labels = [0, 0, 0, 1, 1, 1]
    cal.fit(scores, labels, epochs=300)
    assert cal.calibration_state == CalibrationState.PLATT

    low_p = cal.calibrate(0.15)
    high_p = cal.calibrate(0.85)
    assert 0.0 <= low_p <= 1.0
    assert 0.0 <= high_p <= 1.0
    assert high_p > low_p


def test_calibration_isotonic():
    cal = IsotonicCalibrator()
    scores = [0.1, 0.2, 0.4, 0.5, 0.8, 0.9]
    labels = [0, 0, 1, 0, 1, 1]
    cal.fit(scores, labels)
    assert cal.calibration_state == CalibrationState.ISOTONIC

    p_low = cal.calibrate(0.1)
    p_high = cal.calibrate(0.9)
    assert 0.0 <= p_low <= p_high <= 1.0


def test_calibration_reliability_curve():
    cal = ReliabilityCurveCalibrator(num_bins=5)
    scores = [0.1, 0.15, 0.45, 0.55, 0.85, 0.95]
    labels = [0, 0, 1, 0, 1, 1]
    cal.fit(scores, labels)
    assert cal.calibration_state == CalibrationState.RELIABILITY_CURVE

    p_low = cal.calibrate(0.1)
    p_high = cal.calibrate(0.9)
    assert 0.0 <= p_low <= 1.0
    assert 0.0 <= p_high <= 1.0


def test_uncalibrated_invariant_contract():
    engine_uncalibrated = HypothesisEngine()
    ev = _make_evidence("ev-uncal", prov_id="p-uncal")
    hyp = engine_uncalibrated.evaluate_primary([ev])
    assert hyp is not None
    assert hyp.calibration_state == CalibrationState.NOT_CALIBRATED
    assert hyp.calibrated_probability is None
    assert hyp.raw_score is not None

    with pytest.raises(ValueError, match="calibrated_probability must be None when calibration_state is NOT_CALIBRATED"):
        Hypothesis(
            hypothesis_id="hyp-fail",
            statement=HypothesisStatement.UNVERIFIED_SURFACE_TARGET,
            evidence_ids=("ev-1",),
            supporting_evidence=(ev,),
            refuting_evidence=(),
            score=0.85,
            score_method=ScoreMethod.RULE_ENGINE,
            calibration_state=CalibrationState.NOT_CALIBRATED,
            explanation="Invalid hypothesis",
            provenance=_make_provenance(),
            calibrated_probability=0.85,
        )


def test_hypothesis_engine_with_calibrator():
    cal = PlattCalibrator(a=4.0, b=-2.0)
    engine = HypothesisEngine(calibrator=cal)
    ev = _make_evidence("ev-cal", prov_id="p-cal")
    hyp = engine.evaluate_primary([ev])
    assert hyp is not None
    assert hyp.calibration_state == CalibrationState.PLATT
    assert hyp.calibrated_probability is not None
    assert 0.0 <= hyp.calibrated_probability <= 1.0
    assert "Calibration: State=PLATT" in hyp.detailed_rationale


def test_hypothesis_engine_rationales():
    engine = HypothesisEngine()
    ev_sup = _make_evidence("ev-sup", prov_id="p-sup", references=("sensor=S1_SAR", "target_class=VESSEL"))
    ev_ais = _make_evidence(
        "ev-ref",
        kind=EvidenceKind.AIS_CORRELATION,
        supports=AisCorrelationState.AIS_UNAVAILABLE.value,
        prov_id="p-ref",
    )
    hyp = engine.evaluate_primary([ev_sup, ev_ais])
    assert hyp is not None
    assert hyp.statement == HypothesisStatement.UNVERIFIED_SURFACE_TARGET
    assert "Supporting Evidence (1 items):" in hyp.detailed_rationale
    assert "Refuting Evidence (1 items):" in hyp.detailed_rationale
    assert "sensor=S1_SAR" in hyp.detailed_rationale
    assert "Calibration: Calibrated probability unavailable" in hyp.detailed_rationale


def test_dark_vessel_conjunction_all_scenarios():
    engine = HypothesisEngine()
    maritime_ev = _make_evidence("ev-vessel", prov_id="p-vessel", supports=TargetClass.VESSEL.value)

    ais_down = _make_evidence(
        "ev-ais-down",
        kind=EvidenceKind.AIS_CORRELATION,
        supports=AisCorrelationState.AIS_UNAVAILABLE.value,
        references=("feed_active=False",),
        prov_id="p-down",
    )
    hyp_down = engine.evaluate_primary([maritime_ev, ais_down])
    assert hyp_down is not None
    assert hyp_down.statement == HypothesisStatement.UNVERIFIED_SURFACE_TARGET

    ais_no_cov = _make_evidence(
        "ev-ais-nocov",
        kind=EvidenceKind.AIS_CORRELATION,
        supports=AisCorrelationState.NO_MATCH.value,
        references=("coverage=UNVERIFIED", "feed_active=True"),
        prov_id="p-nocov",
    )
    hyp_nocov = engine.evaluate_primary([maritime_ev, ais_no_cov])
    assert hyp_nocov is not None
    assert hyp_nocov.statement == HypothesisStatement.UNVERIFIED_SURFACE_TARGET

    ais_matched = _make_evidence(
        "ev-ais-match",
        kind=EvidenceKind.AIS_CORRELATION,
        supports=AisCorrelationState.MATCHED.value,
        references=("coverage=VERIFIED", "feed_active=True"),
        prov_id="p-match",
    )
    hyp_matched = engine.evaluate_primary([maritime_ev, ais_matched])
    assert hyp_matched is not None
    assert hyp_matched.statement == HypothesisStatement.UNVERIFIED_SURFACE_TARGET

    ais_verified_nomatch = _make_evidence(
        "ev-ais-dark",
        kind=EvidenceKind.AIS_CORRELATION,
        supports="AIS_NO_MATCH_WITHIN_VERIFIED_COVERAGE",
        references=("coverage=VERIFIED", "feed_active=True"),
        prov_id="p-dark",
    )
    hyp_dark = engine.evaluate_primary([maritime_ev, ais_verified_nomatch])
    assert hyp_dark is not None
    assert hyp_dark.statement == HypothesisStatement.DARK_VESSEL_TRANSITING

    zone_ev = _make_evidence(
        "ev-zone",
        kind=EvidenceKind.SPATIAL_CONTEXT,
        supports=Zone.ZONE_1.value,
        references=("zone=ZONE_1",),
        prov_id="p-zone",
    )
    hyp_zone_only = engine.evaluate_primary([maritime_ev, zone_ev])
    assert hyp_zone_only is not None
    assert hyp_zone_only.statement == HypothesisStatement.UNVERIFIED_SURFACE_TARGET
    assert hyp_zone_only.statement != HypothesisStatement.DARK_VESSEL_TRANSITING
