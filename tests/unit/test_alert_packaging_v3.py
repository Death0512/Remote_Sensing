from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import numpy as np
from PIL import Image
import pytest
import rasterio
import shapely.geometry

from src.alerting.package.builder import AlertPackageBuilder
from src.alerting.priority import determine_alert_priority, validate_priority_gating
from src.alerting.review.store import ReviewAuditRecord, ReviewAuditStore
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


def _create_v3_provenance(prov_id: str = "prov-pkg-v3-01", asset_id: str = "asset-s1-v3-01") -> Provenance:
    return Provenance(
        provenance_id=prov_id,
        source_asset_id=asset_id,
        acquisition_metadata_ref="s3://sentinel1/grd/manifest.safe",
        pipeline_version="3.0.0",
        git_commit="e3b0c44298fc1c149afbf4c8996fb92427ae41e4",
        config_hash="cfg-v3-dynamic-sha256-42",
        model_id="sar-cfar-v3",
        model_version="3.0.0",
        input_checksum="sha256-input-stream-bytes-001",
        output_checksum="sha256-output-stream-bytes-002",
        processing_run_id="run-pkg-v3-900",
        processing_time_utc=datetime(2026, 10, 23, 10, 0, 0, tzinfo=timezone.utc),
    )


def _create_v3_hypothesis(statement: HypothesisStatement = HypothesisStatement.DARK_VESSEL_TRANSITING) -> Hypothesis:
    prov = _create_v3_provenance("prov-hyp-v3", "asset-hyp-v3")
    ev1 = Evidence(
        evidence_id="ev-radar-v3-01",
        kind=EvidenceKind.OBJECT_CONFIDENCE,
        value=0.96,
        quality=0.98,
        source="SarVesselDetectorV3",
        timestamp_utc=datetime(2026, 10, 23, 9, 30, 0, tzinfo=timezone.utc),
        uncertainty=0.04,
        supports="VESSEL",
        references=("snr_db=18.2", "length_m=92.5"),
        provenance=prov,
    )
    ev2 = Evidence(
        evidence_id="ev-ais-v3-01",
        kind=EvidenceKind.AIS_CORRELATION,
        value=0.0,
        quality=1.0,
        source="AisCorrelatorV3",
        timestamp_utc=datetime(2026, 10, 23, 9, 30, 0, tzinfo=timezone.utc),
        uncertainty=0.0,
        supports="NO_MATCH",
        references=("state=NO_MATCH", "coverage=VERIFIED"),
        provenance=prov,
    )
    return Hypothesis(
        hypothesis_id="hyp-v3-100",
        statement=statement,
        evidence_ids=("ev-radar-v3-01", "ev-ais-v3-01"),
        supporting_evidence=(ev1, ev2),
        refuting_evidence=(),
        score=0.97,
        score_method=ScoreMethod.RULE_ENGINE,
        calibration_state=CalibrationState.PLATT,
        explanation="Dark vessel transiting critical maritime zone under verified AIS coverage.",
        provenance=prov,
        detailed_rationale="Conjunction of high-SNR SAR contact and verified lack of transponder ping.",
    )


def _create_v3_detection(prov: Provenance) -> Detection:
    geom = shapely.geometry.box(108.2, 16.1, 108.25, 16.15)
    conf = ConfidenceVector(
        raw_score=0.95,
        model_score=0.95,
        calibrated_probability=0.94,
        uncertainty=0.06,
        quality_score=0.98,
        calibration_state=CalibrationState.PLATT,
    )
    return Detection(
        detection_id="det-v3-001",
        candidate_id="cand-v3-001",
        sensor=SensorType.S1_SAR,
        geom_wgs84=geom,
        geom_px=(120.0, 180.0, 150.0, 210.0),
        crs="EPSG:4326",
        target_class=TargetClass.VESSEL,
        confidence=conf,
        attributes=(("rcs_db", 28.4), ("polarization", "VV")),
        acquisition_time_utc=datetime(2026, 10, 23, 9, 30, 0, tzinfo=timezone.utc),
        provenance=prov,
    )


def test_alert_package_v3_complete_creation(tmp_path: Path) -> None:
    builder = AlertPackageBuilder()
    assert builder.capability == Capability.IMPLEMENTED

    prov = _create_v3_provenance()
    hyp = _create_v3_hypothesis()
    detection = _create_v3_detection(prov)

    alert = Alert(
        alert_id="alt-v3-501",
        hypothesis_id=hyp.hypothesis_id,
        zone=Zone.ZONE_1_CRITICAL,
        priority=AlertPriority.CRITICAL,
        review_state=ReviewState.PENDING_REVIEW,
        bundle_uri=str(tmp_path / "alert_alt-v3-501"),
        provenance=prov,
    )

    raw_matrix = np.random.uniform(0.5, 500.0, size=(48, 48)).astype(np.float32)
    chips = {"raw": raw_matrix}

    pkg_dir = builder.build_package(
        alert=alert,
        hypothesis=hyp,
        chips=chips,
        output_dir=tmp_path,
        detection=detection,
        track_id="trk-v3-001",
    )

    assert pkg_dir.exists()
    assert pkg_dir.is_dir()

    manifest_file = pkg_dir / "manifest.json"
    assert manifest_file.is_file()
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["alert_id"] == "alt-v3-501"
    assert manifest["hypothesis_id"] == "hyp-v3-100"
    assert manifest["schema_version"] == "3.0.0"
    assert manifest["pipeline_version"] == "3.0.0"
    assert manifest["zone"] == Zone.ZONE_1_CRITICAL.value
    assert manifest["priority"] == AlertPriority.CRITICAL.value
    assert manifest["review_state"] == ReviewState.PENDING_REVIEW.value
    assert manifest["bundle_uri"] == str(tmp_path / "alert_alt-v3-501")
    assert manifest["package_format"] == "alert_package.v3"

    chips_manifest = manifest["chips"]
    assert "raw.tif" in chips_manifest["raw"]
    assert "analysis.npy" in chips_manifest["analysis"]
    assert "analysis.tif" in chips_manifest["analysis"]
    assert "preview.png" in chips_manifest["preview"]

    assert manifest["hypothesis"]["hypothesis_id"] == hyp.hypothesis_id
    assert manifest["hypothesis"]["statement"] == hyp.statement.value
    assert manifest["hypothesis"]["score"] == 0.97

    prov_sum = manifest["provenance_summary"]
    assert prov_sum["provenance_id"] == prov.provenance_id
    assert prov_sum["git_commit"] == prov.git_commit
    assert prov_sum["config_hash"] == prov.config_hash
    assert prov_sum["input_checksum"] == prov.input_checksum
    assert prov_sum["output_checksum"] == prov.output_checksum

    geojson_file = pkg_dir / "detection.geojson"
    assert geojson_file.is_file()
    with open(geojson_file, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    assert geojson_data["type"] == "FeatureCollection"
    assert len(geojson_data["features"]) == 1
    feat = geojson_data["features"][0]
    assert feat["type"] == "Feature"
    assert feat["id"] == "det-v3-001"
    assert feat["geometry"]["type"] == "Polygon"
    assert len(feat["bbox"]) == 4

    props = feat["properties"]
    assert props["detection_id"] == "det-v3-001"
    assert props["alert_id"] == "alt-v3-501"
    assert props["hypothesis_id"] == "hyp-v3-100"
    assert props["sensor"] == SensorType.S1_SAR.value
    assert props["target_class"] == TargetClass.VESSEL.value
    assert props["confidence"]["calibrated_probability"] == 0.94
    assert props["attributes"]["rcs_db"] == 28.4
    assert props["track_id"] == "trk-v3-001"

    prov_file = pkg_dir / "provenance.json"
    assert prov_file.is_file()
    with open(prov_file, "r", encoding="utf-8") as f:
        serialized_prov = json.load(f)
    assert serialized_prov["provenance_id"] == prov.provenance_id
    assert serialized_prov["git_commit"] == prov.git_commit
    assert serialized_prov["pipeline_version"] == "3.0.0"
    assert serialized_prov["schema_version"] == "provenance.v3"

    ev_dir = pkg_dir / "evidence"
    assert (ev_dir / "hypothesis.json").is_file()
    assert (ev_dir / "supporting_evidence.json").is_file()
    assert (ev_dir / "refuting_evidence.json").is_file()

    with open(ev_dir / "hypothesis.json", "r", encoding="utf-8") as f:
        hyp_doc = json.load(f)
    assert hyp_doc["hypothesis_id"] == "hyp-v3-100"
    assert hyp_doc["supporting_evidence_count"] == 2
    assert hyp_doc["refuting_evidence_count"] == 0

    with open(ev_dir / "supporting_evidence.json", "r", encoding="utf-8") as f:
        sup_docs = json.load(f)
    assert len(sup_docs) == 2
    assert sup_docs[0]["evidence_id"] == "ev-radar-v3-01"
    assert sup_docs[1]["evidence_id"] == "ev-ais-v3-01"


def test_radiometric_preservation_float32(tmp_path: Path) -> None:
    builder = AlertPackageBuilder()
    prov = _create_v3_provenance()
    hyp = _create_v3_hypothesis()
    alert = Alert(
        alert_id="alt-rad-909",
        hypothesis_id=hyp.hypothesis_id,
        zone=Zone.ZONE_1_CRITICAL,
        priority=AlertPriority.CRITICAL,
        review_state=ReviewState.PENDING_REVIEW,
        bundle_uri=str(tmp_path / "alert_alt-rad-909"),
        provenance=prov,
    )

    float_data = np.array([
        [0.0001234, 15.6789, 450.25],
        [999.9999, 0.000001, 12345.67],
        [3.141592, 2.718281, 100.0],
    ], dtype=np.float32)

    pkg_dir = builder.build_package(
        alert=alert,
        hypothesis=hyp,
        chips={"raw": float_data},
        output_dir=tmp_path,
    )

    raw_tif = pkg_dir / "chips" / "raw" / "raw.tif"
    assert raw_tif.exists()

    with rasterio.open(str(raw_tif)) as src:
        assert src.count == 1
        assert src.dtypes[0] == "float32"
        read_rasterio = src.read(1)
        assert np.array_equal(read_rasterio, float_data)

    pil_loaded = np.array(Image.open(raw_tif))
    assert np.array_equal(pil_loaded, float_data)

    analysis_npy = pkg_dir / "chips" / "analysis" / "analysis.npy"
    analysis_tif = pkg_dir / "chips" / "analysis" / "analysis.tif"
    assert analysis_npy.exists()
    assert analysis_tif.exists()

    loaded_analysis = np.load(analysis_npy)
    assert loaded_analysis.dtype == np.float32
    assert float(np.min(loaded_analysis)) >= 0.0
    assert float(np.max(loaded_analysis)) <= 1.0

    preview_png = pkg_dir / "chips" / "preview" / "preview.png"
    assert preview_png.exists()
    with Image.open(preview_png) as img:
        assert img.format == "PNG"
        prev_arr = np.array(img)
        assert prev_arr.dtype == np.uint8


def test_immutable_audit_ledger_sequence_and_hash_chain(tmp_path: Path) -> None:
    db_file = tmp_path / "audit_v3.db"
    jsonl_file = tmp_path / "audit_v3.jsonl"

    store = ReviewAuditStore(db_path=db_file, journal_path=jsonl_file)
    assert store.capability == Capability.IMPLEMENTED

    rec1 = store.record_review(
        alert_id="alt-01",
        action=ReviewAction.FLAG_UNCERTAIN,
        operator_id="op_alpha",
        notes="First review: uncertain classification",
    )
    assert rec1.id == 1
    assert rec1.previous_hash == "GENESIS"
    assert rec1.previous_state == ReviewState.PENDING_REVIEW
    assert rec1.new_state == ReviewState.FLAGGED_UNCERTAIN

    rec2 = store.record_review(
        alert_id="alt-01",
        action=ReviewAction.CONFIRM,
        operator_id="op_bravo",
        notes="Second review: confirmed after optical check",
    )
    assert rec2.id == 2
    assert rec2.previous_hash == rec1.entry_hash
    assert rec2.previous_state == ReviewState.FLAGGED_UNCERTAIN
    assert rec2.new_state == ReviewState.CONFIRMED

    rec3 = store.record_review(
        alert_id="alt-02",
        action=ReviewAction.DISMISS,
        operator_id="op_charlie",
        notes="Clutter dismissal",
    )
    assert rec3.id == 3
    assert rec3.previous_hash == rec2.entry_hash
    assert rec3.previous_state == ReviewState.PENDING_REVIEW
    assert rec3.new_state == ReviewState.DISMISSED

    assert store.verify_integrity() is True
    store.close()


def test_audit_ledger_append_only_triggers_and_tamper_detection(tmp_path: Path) -> None:
    db_file = tmp_path / "tamper_v3.db"
    store = ReviewAuditStore(db_path=db_file)

    store.record_review(
        alert_id="alt-t1",
        action=ReviewAction.CONFIRM,
        operator_id="analyst_1",
        notes="Approved",
    )
    store.record_review(
        alert_id="alt-t2",
        action=ReviewAction.DISMISS,
        operator_id="analyst_2",
        notes="Declined",
    )

    assert store.verify_integrity() is True

    raw_conn = sqlite3.connect(str(db_file))

    with pytest.raises(sqlite3.DatabaseError) as exc_update:
        raw_conn.execute("UPDATE review_audit_log SET notes = 'tampered' WHERE id = 1")
    assert "Audit log is append-only" in str(exc_update.value)

    with pytest.raises(sqlite3.DatabaseError) as exc_delete:
        raw_conn.execute("DELETE FROM review_audit_log WHERE id = 2")
    assert "Audit log is append-only" in str(exc_delete.value)

    raw_conn.execute("DROP TRIGGER prevent_audit_update")
    raw_conn.execute("UPDATE review_audit_log SET notes = 'illicit modification' WHERE id = 1")
    raw_conn.commit()
    raw_conn.close()

    assert store.verify_integrity() is False
    store.close()


def test_dual_write_race_elimination_and_archive_export(tmp_path: Path) -> None:
    db_file = tmp_path / "concurrent_audit.db"
    archive_file = tmp_path / "export_audit.jsonl"

    store = ReviewAuditStore(db_path=db_file, journal_path=tmp_path / "stream.jsonl")

    num_alerts = 30

    def worker(index: int) -> ReviewAuditRecord:
        action = ReviewAction.CONFIRM if index % 2 == 0 else ReviewAction.DISMISS
        return store.record_review(
            alert_id=f"alt-concurrent-{index}",
            action=action,
            operator_id=f"operator-{index}",
            notes=f"Concurrent entry {index}",
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(worker, range(num_alerts)))

    assert len(records) == num_alerts
    assert store.verify_integrity() is True

    history = store.get_review_history()
    assert len(history) == num_alerts

    ids = [r.id for r in history]
    assert ids == list(range(1, num_alerts + 1))

    for i in range(1, len(history)):
        assert history[i].previous_hash == history[i - 1].entry_hash

    exported_path = store.export_archive(archive_file)
    assert exported_path.exists()

    with open(archive_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == num_alerts
    assert lines[0]["id"] == 1
    assert lines[-1]["id"] == num_alerts
    assert lines[-1]["entry_hash"] == history[-1].entry_hash

    with pytest.raises(ValueError):
        store.record_review(
            alert_id="alt-concurrent-0",
            action=ReviewAction.DISMISS,
            operator_id="operator-duplicate",
        )

    store.close()


def test_cli_packaging_error_not_swallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.cli.main import cli_entrypoint
    from tests.mocks.mock_data import generate_synthetic_sar_scene

    sar_scene = generate_synthetic_sar_scene(
        height=100,
        width=100,
        targets=[(30, 30, 4, 4, 300.0)],
        seed=123,
    )
    img_path = tmp_path / "sar_scene.npy"
    np.save(str(img_path), sar_scene)

    def mock_build_package(*args, **kwargs):
        raise OSError("Disk write failure in package builder")

    monkeypatch.setattr(AlertPackageBuilder, "build_package", mock_build_package)

    with pytest.raises(RuntimeError) as exc_info:
        cli_entrypoint([
            "scan",
            "--image", str(img_path),
            "--sensor", "S1_GRD",
            "--output-dir", str(tmp_path / "out"),
        ])
    assert "Failed to generate alert package" in str(exc_info.value)

