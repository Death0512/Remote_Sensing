from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import numpy as np
from PIL import Image
import pytest
import shapely.geometry

from src.alerting.package.builder import AlertPackageBuilder
from src.alerting.priority import determine_alert_priority, validate_priority_gating
from src.alerting.review.store import ReviewAuditStore
from src.domain.alert import Alert
from src.domain.confidence import ConfidenceVector
from src.domain.detection import Detection
from src.domain.enums import (
    AlertPriority,
    CalibrationState,
    Capability,
    EvidenceKind,
    HypothesisStatement,
    ReviewAction,
    ReviewState,
    ScoreMethod,
    SensorType,
    TargetClass,
    Zone,
)
from src.domain.evidence import Evidence
from src.domain.hypothesis import Hypothesis
from src.domain.provenance import Provenance


def _make_provenance(prov_id: str = "prov-pkg-01", asset_id: str = "asset-s1-01") -> Provenance:
    return Provenance(
        provenance_id=prov_id,
        source_asset_id=asset_id,
        acquisition_metadata_ref="s3://imagery/sentinel1/meta.safe",
        pipeline_version="2.0.0",
        git_commit="30d2bc8",
        config_hash="cfg-pkg-hash-42",
        model_id="detector-cfar-v2",
        model_version="2.1.0",
        input_checksum="input-sha-123",
        output_checksum="output-sha-456",
        processing_run_id="run-pkg-789",
        processing_time_utc=datetime(2026, 10, 22, 14, 0, 0, tzinfo=timezone.utc),
    )


def _make_hypothesis(statement: HypothesisStatement = HypothesisStatement.DARK_VESSEL_TRANSITING) -> Hypothesis:
    prov = _make_provenance("prov-hyp-01", "asset-hyp-01")
    ev1 = Evidence(
        evidence_id="ev-radar-01",
        kind=EvidenceKind.OBJECT_CONFIDENCE,
        value=0.91,
        quality=0.95,
        source="SarVesselDetector",
        timestamp_utc=datetime(2026, 10, 22, 13, 30, 0, tzinfo=timezone.utc),
        uncertainty=0.09,
        supports="VESSEL",
        references=("snr_db=16.5", "length_m=84.0"),
        provenance=prov,
    )
    ev2 = Evidence(
        evidence_id="ev-ais-01",
        kind=EvidenceKind.AIS_CORRELATION,
        value=0.0,
        quality=1.0,
        source="AisCorrelator",
        timestamp_utc=datetime(2026, 10, 22, 13, 30, 0, tzinfo=timezone.utc),
        uncertainty=0.0,
        supports="NO_MATCH",
        references=("state=NO_MATCH", "coverage=VERIFIED"),
        provenance=prov,
    )
    return Hypothesis(
        hypothesis_id="hyp-001",
        statement=statement,
        evidence_ids=("ev-radar-01", "ev-ais-01"),
        supporting_evidence=(ev1, ev2),
        refuting_evidence=(),
        score=0.93,
        score_method=ScoreMethod.RULE_ENGINE,
        calibration_state=CalibrationState.PLATT,
        explanation="Verified maritime target without active transponder under full receiver coverage.",
        provenance=prov,
        detailed_rationale="Conjunction of high-confidence radar detection and confirmed absence of AIS broadcast.",
    )


def _make_detection(prov: Provenance) -> Detection:
    geom = shapely.geometry.box(108.1, 15.2, 108.105, 15.205)
    conf = ConfidenceVector(
        raw_score=0.91,
        model_score=0.91,
        calibrated_probability=0.89,
        uncertainty=0.11,
        quality_score=0.95,
        calibration_state=CalibrationState.PLATT,
    )
    return Detection(
        detection_id="det-001",
        candidate_id="cand-001",
        sensor=SensorType.S1_SAR,
        geom_wgs84=geom,
        geom_px=(100.0, 150.0, 120.0, 175.0),
        crs="EPSG:4326",
        target_class=TargetClass.VESSEL,
        confidence=conf,
        attributes=(("rcs_db", 24.5), ("polarization", "VV")),
        acquisition_time_utc=datetime(2026, 10, 22, 13, 30, 0, tzinfo=timezone.utc),
        provenance=prov,
    )


def test_alert_package_directory_structure(tmp_path: Path) -> None:
    builder = AlertPackageBuilder()
    assert builder.capability == Capability.IMPLEMENTED

    prov = _make_provenance()
    hyp = _make_hypothesis(HypothesisStatement.DARK_VESSEL_TRANSITING)
    alert = Alert(
        alert_id="alt-402",
        hypothesis_id=hyp.hypothesis_id,
        zone=Zone.ZONE_1_CRITICAL,
        priority=AlertPriority.CRITICAL,
        review_state=ReviewState.PENDING_REVIEW,
        bundle_uri=str(tmp_path / "alert_alt-402"),
        provenance=prov,
    )

    raw_chip = np.random.uniform(0.01, 100.0, size=(64, 64)).astype(np.float32)
    chips = {"raw": raw_chip}

    detection = _make_detection(prov)
    pkg_path = builder.build_package(
        alert=alert,
        hypothesis=hyp,
        chips=chips,
        output_dir=tmp_path,
        detection=detection,
        track_id="trk-999",
    )

    assert pkg_path.exists()
    assert pkg_path.is_dir()
    assert (pkg_path / "manifest.json").is_file()
    assert (pkg_path / "detection.geojson").is_file()
    assert (pkg_path / "provenance.json").is_file()
    assert (pkg_path / "evidence" / "hypothesis.json").is_file()
    assert (pkg_path / "evidence" / "supporting_evidence.json").is_file()
    assert (pkg_path / "evidence" / "refuting_evidence.json").is_file()
    assert (pkg_path / "chips" / "raw").is_dir()
    assert (pkg_path / "chips" / "analysis").is_dir()
    assert (pkg_path / "chips" / "preview").is_dir()
    assert any((pkg_path / "chips" / "raw").iterdir())
    assert any((pkg_path / "chips" / "analysis").iterdir())
    assert any((pkg_path / "chips" / "preview").iterdir())


def test_provenance_and_manifest_serialization(tmp_path: Path) -> None:
    builder = AlertPackageBuilder()
    prov = _make_provenance()
    hyp = _make_hypothesis()
    alert = Alert(
        alert_id="alt-403",
        hypothesis_id=hyp.hypothesis_id,
        zone=Zone.ZONE_1,
        priority=AlertPriority.CRITICAL,
        review_state=ReviewState.PENDING_REVIEW,
        bundle_uri=str(tmp_path / "alert_alt-403"),
        provenance=prov,
    )
    raw_chip = np.linspace(0.0, 25.0, 1600, dtype=np.float32).reshape(40, 40)
    chips = {"raw": raw_chip}

    pkg_path = builder.build_package(
        alert=alert,
        hypothesis=hyp,
        chips=chips,
        output_dir=tmp_path,
    )

    with open(pkg_path / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["alert_id"] == "alt-403"
    assert manifest["hypothesis_id"] == "hyp-001"
    assert manifest["schema_version"] == "2.0.0"
    assert manifest["pipeline_version"] == "2.0.0"
    assert manifest["zone"] == Zone.ZONE_1.value
    assert manifest["priority"] == AlertPriority.CRITICAL.value
    assert manifest["review_state"] == ReviewState.PENDING_REVIEW.value
    assert manifest["operator_review_status"] == ReviewState.PENDING_REVIEW.value
    assert manifest["provenance_id"] == prov.provenance_id
    assert manifest["hypothesis"]["statement"] == hyp.statement.value
    assert manifest["hypothesis"]["explanation"] == hyp.explanation

    with open(pkg_path / "provenance.json", "r", encoding="utf-8") as f:
        prov_dict = json.load(f)
    assert prov_dict["provenance_id"] == prov.provenance_id
    assert prov_dict["source_asset_id"] == prov.source_asset_id
    assert prov_dict["acquisition_metadata_ref"] == prov.acquisition_metadata_ref
    assert prov_dict["pipeline_version"] == prov.pipeline_version
    assert prov_dict["git_commit"] == prov.git_commit
    assert prov_dict["config_hash"] == prov.config_hash
    assert prov_dict["model_id"] == prov.model_id
    assert prov_dict["model_version"] == prov.model_version
    assert prov_dict["input_checksum"] == prov.input_checksum
    assert prov_dict["output_checksum"] == prov.output_checksum
    assert prov_dict["processing_run_id"] == prov.processing_run_id
    assert prov_dict["processing_time_utc"] == prov.processing_time_utc.isoformat()

    with open(pkg_path / "evidence" / "supporting_evidence.json", "r", encoding="utf-8") as f:
        ev_list = json.load(f)
    assert len(ev_list) == len(hyp.supporting_evidence)
    assert ev_list[0]["evidence_id"] == hyp.supporting_evidence[0].evidence_id
    assert ev_list[0]["kind"] == hyp.supporting_evidence[0].kind.value


def test_raw_chip_preserves_radiometry(tmp_path: Path) -> None:
    builder = AlertPackageBuilder()
    prov = _make_provenance()
    hyp = _make_hypothesis()
    alert = Alert(
        alert_id="alt-radiometry",
        hypothesis_id=hyp.hypothesis_id,
        zone=Zone.ZONE_1,
        priority=AlertPriority.CRITICAL,
        review_state=ReviewState.PENDING,
        bundle_uri=str(tmp_path / "alert_alt-radiometry"),
        provenance=prov,
    )
    raw_matrix = np.array([
        [0.00341, 14.8821, 102.44],
        [0.10022, 5.5555, 999.123],
    ], dtype=np.float32)

    pkg_path = builder.build_package(
        alert=alert,
        hypothesis=hyp,
        chips={"raw": raw_matrix},
        output_dir=tmp_path,
    )

    raw_file = pkg_path / "chips" / "raw" / "raw.tif"
    assert raw_file.exists()
    loaded_arr = np.array(Image.open(raw_file))
    assert np.array_equal(raw_matrix, loaded_arr)


def test_review_audit_state_transitions(tmp_path: Path) -> None:
    db_file = tmp_path / "audit.db"
    store = ReviewAuditStore(db_path=db_file)
    assert store.capability == Capability.IMPLEMENTED

    alert_1 = "alt-rev-01"
    assert store.get_review_state(alert_1) == ReviewState.PENDING_REVIEW

    t1 = datetime(2026, 10, 22, 15, 0, 0, tzinfo=timezone.utc)
    rec_confirm = store.record_review(
        alert_id=alert_1,
        action=ReviewAction.CONFIRM,
        operator_id="operator_01",
        notes="Vessel confirmed by SAR-optical correlation",
        timestamp_utc=t1,
    )
    assert rec_confirm.alert_id == alert_1
    assert rec_confirm.previous_state == ReviewState.PENDING_REVIEW
    assert rec_confirm.new_state == ReviewState.CONFIRMED
    assert rec_confirm.operator_id == "operator_01"
    assert rec_confirm.notes == "Vessel confirmed by SAR-optical correlation"
    assert rec_confirm.timestamp_utc == t1
    assert store.get_review_state(alert_1) == ReviewState.CONFIRMED

    alert_2 = "alt-rev-02"
    t2 = datetime(2026, 10, 22, 15, 5, 0, tzinfo=timezone.utc)
    rec_dismiss = store.record_review(
        alert_id=alert_2,
        action=ReviewAction.DISMISS,
        operator_id="operator_02",
        notes="Dismissed as sea clutter artifact",
        timestamp_utc=t2,
    )
    assert rec_dismiss.alert_id == alert_2
    assert rec_dismiss.previous_state == ReviewState.PENDING_REVIEW
    assert rec_dismiss.new_state == ReviewState.DISMISSED
    assert rec_dismiss.notes == "Dismissed as sea clutter artifact"
    assert store.get_review_state(alert_2) == ReviewState.DISMISSED

    alert_3 = "alt-rev-03"
    rec_uncertain = store.record_review(
        alert_id=alert_3,
        action=ReviewAction.FLAG_UNCERTAIN,
        operator_id="operator_03",
        notes="Awaiting high-resolution optical pass",
    )
    assert rec_uncertain.new_state == ReviewState.FLAGGED_UNCERTAIN
    assert store.get_review_state(alert_3) == ReviewState.FLAGGED_UNCERTAIN

    with open(tmp_path / "audit.jsonl", "r", encoding="utf-8") as f:
        journal_lines = [json.loads(line) for line in f if line.strip()]
    assert len(journal_lines) == 3
    assert journal_lines[0]["alert_id"] == alert_1
    assert journal_lines[0]["action"] == ReviewAction.CONFIRM.value
    assert journal_lines[1]["alert_id"] == alert_2
    assert journal_lines[1]["action"] == ReviewAction.DISMISS.value


def test_tamper_resistant_audit_log(tmp_path: Path) -> None:
    db_file = tmp_path / "tamper_test.db"
    store = ReviewAuditStore(db_path=db_file)

    rec1 = store.record_review(
        alert_id="alt-sec-01",
        action=ReviewAction.CONFIRM,
        operator_id="analyst_1",
        notes="Initial review",
    )
    rec2 = store.record_review(
        alert_id="alt-sec-02",
        action=ReviewAction.DISMISS,
        operator_id="analyst_2",
        notes="Initial review",
    )

    assert rec1.id == 1
    assert rec2.id == 2
    assert rec1.previous_hash == "GENESIS"
    assert rec2.previous_hash == rec1.entry_hash
    assert store.verify_integrity() is True

    with pytest.raises(ValueError):
        store.record_review(
            alert_id="alt-sec-01",
            action=ReviewAction.DISMISS,
            operator_id="attacker_1",
            notes="Attempting modification of confirmed alert",
        )

    raw_conn = sqlite3.connect(str(db_file))
    with pytest.raises(sqlite3.DatabaseError) as exc_info:
        raw_conn.execute("UPDATE review_audit_log SET notes = 'tampered' WHERE id = 1")
    assert "Audit log is append-only" in str(exc_info.value)

    with pytest.raises(sqlite3.DatabaseError) as exc_info_del:
        raw_conn.execute("DELETE FROM review_audit_log WHERE id = 1")
    assert "Audit log is append-only" in str(exc_info_del.value)

    raw_conn.close()
    assert store.verify_integrity() is True


def test_priority_gating_critical_zone_1() -> None:
    dark_vessel = HypothesisStatement.DARK_VESSEL_TRANSITING
    incursion = HypothesisStatement.UNIDENTIFIED_AIR_INCURSION
    loitering = HypothesisStatement.SURFACE_VESSEL_LOITERING

    p_crit_1 = determine_alert_priority(Zone.ZONE_1_CRITICAL, dark_vessel)
    assert p_crit_1 == AlertPriority.CRITICAL

    p_crit_2 = determine_alert_priority(Zone.ZONE_1, incursion)
    assert p_crit_2 == AlertPriority.CRITICAL

    p_high = determine_alert_priority(Zone.ZONE_1, loitering)
    assert p_high == AlertPriority.HIGH

    p_z2_crit = determine_alert_priority(Zone.ZONE_2, dark_vessel)
    assert p_z2_crit == AlertPriority.HIGH

    p_z2_other = determine_alert_priority(Zone.ZONE_2, loitering)
    assert p_z2_other == AlertPriority.MEDIUM

    p_z3_crit = determine_alert_priority(Zone.ZONE_3, dark_vessel)
    assert p_z3_crit == AlertPriority.MEDIUM

    p_z3_other = determine_alert_priority(Zone.ZONE_3, loitering)
    assert p_z3_other == AlertPriority.LOW

    p_outside = determine_alert_priority(Zone.OUTSIDE_AOI, dark_vessel)
    assert p_outside == AlertPriority.INFORMATIONAL

    validate_priority_gating(AlertPriority.CRITICAL, Zone.ZONE_1, dark_vessel)

    with pytest.raises(ValueError):
        validate_priority_gating(AlertPriority.CRITICAL, Zone.ZONE_2, dark_vessel)

    with pytest.raises(ValueError):
        validate_priority_gating(AlertPriority.CRITICAL, Zone.ZONE_1, loitering)
