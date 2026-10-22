from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import pytest
from shapely.geometry import Point, Polygon
from src.domain import (
    AisCorrelationState,
    Alert,
    AlertPriority,
    CalibrationState,
    Candidate,
    Capability,
    ConfidenceVector,
    Detection,
    Evidence,
    EvidenceKind,
    GeometricState,
    Hypothesis,
    HypothesisStatement,
    Observation,
    ProcessedObservation,
    Provenance,
    QualityTier,
    RadiometricState,
    ReviewState,
    ScoreMethod,
    SensorType,
    TargetClass,
    Track,
    TrackLifecycle,
    Zone,
)


def _sample_provenance() -> Provenance:
    return Provenance(
        provenance_id="prov-001",
        source_asset_id="asset-001",
        acquisition_metadata_ref="meta-001",
        pipeline_version="2.0.0",
        git_commit="abcdef1",
        config_hash="hash123",
        model_id=None,
        model_version=None,
        input_checksum="inchk1",
        output_checksum="outchk1",
        processing_run_id="run-001",
        processing_time_utc=datetime(2026, 10, 22, 0, 0, 0, tzinfo=timezone.utc),
    )


def test_capability_truth() -> None:
    expected = {
        "IMPLEMENTED",
        "PARTIAL",
        "HEURISTIC",
        "EXTERNAL_MODEL",
        "MOCK",
        "UNAVAILABLE",
    }
    actual = {c.value for c in Capability}
    assert actual == expected
    for name in expected:
        assert getattr(Capability, name).name == name


def test_no_fabricated_probability() -> None:
    vec_default = ConfidenceVector(raw_score=0.8)
    assert vec_default.raw_score == 0.8
    assert vec_default.calibrated_probability is None
    assert vec_default.calibration_state == CalibrationState.NOT_CALIBRATED

    with pytest.raises(ValueError, match="calibrated_probability must be None"):
        ConfidenceVector(
            raw_score=0.8,
            calibrated_probability=0.8,
            calibration_state=CalibrationState.NOT_CALIBRATED,
        )

    vec_calibrated = ConfidenceVector(
        raw_score=0.8,
        calibrated_probability=0.75,
        calibration_state=CalibrationState.PLATT,
    )
    assert vec_calibrated.calibrated_probability == 0.75
    assert vec_calibrated.calibration_state == CalibrationState.PLATT


def test_zone_enum() -> None:
    expected = {"OUTSIDE_AOI", "ZONE_1", "ZONE_2", "ZONE_3"}
    actual = {z.value for z in Zone}
    assert actual == expected
    assert Zone.OUTSIDE_AOI.value == "OUTSIDE_AOI"
    assert Zone.ZONE_1.value == "ZONE_1"
    assert Zone.ZONE_2.value == "ZONE_2"
    assert Zone.ZONE_3.value == "ZONE_3"


def test_provenance_immutability() -> None:
    prov = _sample_provenance()
    with pytest.raises(FrozenInstanceError):
        prov.git_commit = "mutation"

    vec = ConfidenceVector(raw_score=0.5)
    with pytest.raises(FrozenInstanceError):
        vec.raw_score = 0.9

    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    dt = datetime(2026, 10, 22, 1, 0, 0, tzinfo=timezone.utc)

    obs = Observation(
        observation_id="obs-1",
        source="CDSE",
        platform="Sentinel-1A",
        sensor=SensorType.S1_SAR,
        product="GRD",
        processing_level="LEVEL1",
        acquisition_time_utc=dt,
        crs="EPSG:4326",
        ground_sample_distance_m=10.0,
        bounds_wgs84=poly,
        asset_uri="s3://bucket/asset.safe",
        metadata_uri="s3://bucket/meta.xml",
        quality_flags=("OK",),
        provenance=prov,
    )
    with pytest.raises(FrozenInstanceError):
        obs.crs = "EPSG:3857"

    proc = ProcessedObservation(
        processed_id="proc-1",
        observation_id="obs-1",
        preprocessing_profile="sar_standard",
        radiometric_state=RadiometricState.CALIBRATED_SIGMA0,
        geometric_state=GeometricState.TERRAIN_CORRECTED,
        masks=(("land", "uri1"),),
        quality_tier=QualityTier.DATA_OK,
        provenance=prov,
    )
    with pytest.raises(FrozenInstanceError):
        proc.quality_tier = QualityTier.STALE

    cand = Candidate(
        candidate_id="cand-1",
        processed_id="proc-1",
        geom_px=(10.0, 10.0, 20.0, 20.0),
        geom_wgs84=Point(108.0, 16.0),
        generator="cfar",
        generator_capability=Capability.IMPLEMENTED,
        features=(("snr", 15.0),),
        quality_flags=(),
        provenance=prov,
    )
    with pytest.raises(FrozenInstanceError):
        cand.generator = "other"

    det = Detection(
        detection_id="det-1",
        candidate_id="cand-1",
        sensor=SensorType.S1_SAR,
        geom_wgs84=Point(108.0, 16.0),
        geom_px=(10.0, 10.0, 20.0, 20.0),
        crs="EPSG:4326",
        target_class=TargetClass.VESSEL,
        confidence=vec,
        attributes=(("length_m", 50.0),),
        acquisition_time_utc=dt,
        provenance=prov,
    )
    with pytest.raises(FrozenInstanceError):
        det.crs = "EPSG:3857"

    trk = Track(
        track_id="trk-1",
        detection_ids=("det-1",),
        state_estimates=((("x", 1.0), ("y", 2.0)),),
        track_confidence=0.8,
        lifecycle=TrackLifecycle.CONFIRMED,
        provenance=prov,
    )
    with pytest.raises(FrozenInstanceError):
        trk.track_confidence = 0.1

    ev = Evidence(
        evidence_id="ev-1",
        kind=EvidenceKind.OBJECT_CONFIDENCE,
        value=0.85,
        quality=1.0,
        source="sar_cfar",
        timestamp_utc=dt,
        uncertainty=0.05,
        supports="target_present",
        references=("det-1",),
        provenance=prov,
    )
    with pytest.raises(FrozenInstanceError):
        ev.value = 0.2

    hyp = Hypothesis(
        hypothesis_id="hyp-1",
        statement=HypothesisStatement.DARK_VESSEL_CANDIDATE,
        evidence_ids=("ev-1",),
        supporting_evidence=("ev-1",),
        refuting_evidence=(),
        score=0.85,
        score_method=ScoreMethod.RULE_ENGINE,
        calibration_state=CalibrationState.NOT_CALIBRATED,
        explanation="Valid SAR detection without matching AIS signal.",
        provenance=prov,
    )
    with pytest.raises(FrozenInstanceError):
        hyp.score = 0.0

    alert = Alert(
        alert_id="alt-1",
        hypothesis_id="hyp-1",
        zone=Zone.ZONE_1,
        priority=AlertPriority.HIGH,
        review_state=ReviewState.PENDING,
        bundle_uri="s3://bucket/alerts/alt-1",
        provenance=prov,
    )
    with pytest.raises(FrozenInstanceError):
        alert.priority = AlertPriority.CRITICAL


def test_hypothesis_requires_explanation_and_evidence() -> None:
    prov = _sample_provenance()

    with pytest.raises(ValueError, match="Hypothesis requires non-empty explanation"):
        Hypothesis(
            hypothesis_id="hyp-1",
            statement=HypothesisStatement.DARK_VESSEL_CANDIDATE,
            evidence_ids=("ev-1",),
            supporting_evidence=("ev-1",),
            refuting_evidence=(),
            score=0.85,
            score_method=ScoreMethod.RULE_ENGINE,
            calibration_state=CalibrationState.NOT_CALIBRATED,
            explanation="",
            provenance=prov,
        )

    with pytest.raises(ValueError, match="Hypothesis requires non-empty explanation"):
        Hypothesis(
            hypothesis_id="hyp-1",
            statement=HypothesisStatement.DARK_VESSEL_CANDIDATE,
            evidence_ids=("ev-1",),
            supporting_evidence=("ev-1",),
            refuting_evidence=(),
            score=0.85,
            score_method=ScoreMethod.RULE_ENGINE,
            calibration_state=CalibrationState.NOT_CALIBRATED,
            explanation="   ",
            provenance=prov,
        )

    with pytest.raises(ValueError, match="Hypothesis requires at least one evidence item"):
        Hypothesis(
            hypothesis_id="hyp-1",
            statement=HypothesisStatement.DARK_VESSEL_CANDIDATE,
            evidence_ids=(),
            supporting_evidence=(),
            refuting_evidence=(),
            score=0.85,
            score_method=ScoreMethod.RULE_ENGINE,
            calibration_state=CalibrationState.NOT_CALIBRATED,
            explanation="Valid explanation",
            provenance=prov,
        )

    valid_hyp = Hypothesis(
        hypothesis_id="hyp-1",
        statement=HypothesisStatement.DARK_VESSEL_CANDIDATE,
        evidence_ids=("ev-1",),
        supporting_evidence=("ev-1",),
        refuting_evidence=(),
        score=0.85,
        score_method=ScoreMethod.RULE_ENGINE,
        calibration_state=CalibrationState.NOT_CALIBRATED,
        explanation="Valid non-empty explanation",
        provenance=prov,
    )
    assert valid_hyp.explanation == "Valid non-empty explanation"
    assert valid_hyp.evidence_ids == ("ev-1",)


def test_alert_contract() -> None:
    prov = _sample_provenance()
    alert = Alert(
        alert_id="alt-100",
        hypothesis_id="hyp-100",
        zone=Zone.ZONE_1,
        priority=AlertPriority.CRITICAL,
        review_state=ReviewState.PENDING,
        bundle_uri="file:///data/alerts/alt-100",
        provenance=prov,
    )
    assert alert.alert_id == "alt-100"
    assert alert.zone == Zone.ZONE_1
    assert alert.priority == AlertPriority.CRITICAL
    assert alert.review_state == ReviewState.PENDING
    assert alert.bundle_uri == "file:///data/alerts/alt-100"
    assert alert.provenance == prov

    alert_from_strings = Alert(
        alert_id="alt-101",
        hypothesis_id="hyp-101",
        zone="ZONE_2",
        priority="HIGH",
        review_state="CONFIRMED",
        bundle_uri="file:///data/alerts/alt-101",
        provenance=prov,
    )
    assert alert_from_strings.zone == Zone.ZONE_2
    assert alert_from_strings.priority == AlertPriority.HIGH
    assert alert_from_strings.review_state == ReviewState.CONFIRMED


def test_provenance_required() -> None:
    with pytest.raises(TypeError):
        Alert(
            alert_id="alt-1",
            hypothesis_id="hyp-1",
            zone=Zone.ZONE_1,
            priority=AlertPriority.HIGH,
            review_state=ReviewState.PENDING,
            bundle_uri="/path",
        )
