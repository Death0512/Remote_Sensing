from datetime import datetime, timezone
import json
from pathlib import Path
import pytest
from shapely.geometry import Polygon, box

from src.domain.confidence import ConfidenceVector
from src.domain.detection import Detection
from src.domain.enums import Capability, SensorType, TargetClass
from src.domain.provenance import Provenance
from src.evaluation.benchmarking.metrics import BenchmarkEvaluator, BenchmarkReport
from src.evaluation.benchmarking.splits import LeakageError, SplitValidator


def _make_provenance(prov_id: str = "prov-eval-01") -> Provenance:
    return Provenance(
        provenance_id=prov_id,
        source_asset_id="asset-eval-01",
        acquisition_metadata_ref="meta://eval",
        pipeline_version="2.0.0",
        git_commit="30d2bc8",
        config_hash="cfg-eval",
        model_id=None,
        model_version=None,
        input_checksum="inchk",
        output_checksum="outchk",
        processing_run_id="run-eval",
        processing_time_utc=datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc),
    )


def _make_detection(
    det_id: str,
    bbox: tuple[float, float, float, float],
    target_class: TargetClass = TargetClass.VESSEL,
    overall_conf: float = 0.9,
) -> Detection:
    geom = box(bbox[0], bbox[1], bbox[2], bbox[3])
    return Detection(
        detection_id=det_id,
        candidate_id=f"cand-{det_id}",
        sensor=SensorType.S1_SAR,
        geom_wgs84=geom,
        geom_px=(0.0, 0.0, 10.0, 10.0),
        crs="EPSG:4326",
        target_class=target_class,
        confidence=ConfidenceVector(
            raw_score=overall_conf,
            model_score=overall_conf,
            uncertainty=0.1,
            quality_score=1.0,
        ),
        attributes=(("dummy", "val"),),
        acquisition_time_utc=datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc),
        provenance=_make_provenance(),
    )


def test_spatial_leakage_detected() -> None:
    validator = SplitValidator()
    assert validator.capability == Capability.IMPLEMENTED

    train_aois = [(108.0, 15.0, 109.0, 16.0)]
    test_aois = [(108.5, 15.5, 109.5, 16.5)]
    train_times = [
        (
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
    ]
    test_times = [
        (
            datetime(2026, 6, 1, tzinfo=timezone.utc),
            datetime(2026, 9, 1, tzinfo=timezone.utc),
        )
    ]

    with pytest.raises(LeakageError, match="Spatial leakage detected"):
        validator.validate_splits(train_aois, test_aois, train_times, test_times)


def test_temporal_leakage_detected() -> None:
    validator = SplitValidator()

    train_aois = [(100.0, 10.0, 101.0, 11.0)]
    test_aois = [(110.0, 20.0, 111.0, 21.0)]
    train_times = [
        (
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
    ]
    test_times = [
        (
            datetime(2026, 5, 1, tzinfo=timezone.utc),
            datetime(2026, 12, 1, tzinfo=timezone.utc),
        )
    ]

    with pytest.raises(LeakageError, match="Temporal leakage detected"):
        validator.validate_splits(train_aois, test_aois, train_times, test_times)


def test_disjoint_splits_pass_validation() -> None:
    validator = SplitValidator()

    train_aois = [box(100.0, 10.0, 101.0, 11.0)]
    test_aois = [box(102.0, 10.0, 103.0, 11.0)]
    train_times = [
        (
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
    ]
    test_times = [
        (
            datetime(2026, 4, 1, tzinfo=timezone.utc),
            datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
    ]

    result = validator.validate_splits(train_aois, test_aois, train_times, test_times)
    assert result is True


def test_iou_matching_and_metrics_calculation(tmp_path: Path) -> None:
    evaluator = BenchmarkEvaluator(iou_threshold=0.5, pipeline_version="2.0.0", git_commit="30d2bc8")
    assert evaluator.capability == Capability.IMPLEMENTED

    gt_1 = _make_detection("gt-1", (0.0, 0.0, 2.0, 2.0), TargetClass.VESSEL)
    gt_2 = _make_detection("gt-2", (5.0, 5.0, 7.0, 7.0), TargetClass.AIRCRAFT)
    gt_3 = _make_detection("gt-3", (10.0, 10.0, 12.0, 12.0), TargetClass.VESSEL)
    ground_truth = [gt_1, gt_2, gt_3]

    pred_1 = _make_detection("pred-1", (0.0, 0.0, 2.0, 2.0), TargetClass.VESSEL, overall_conf=0.95)
    pred_2 = _make_detection("pred-2", (5.2, 5.2, 7.0, 7.0), TargetClass.AIRCRAFT, overall_conf=0.85)
    pred_3 = _make_detection("pred-3", (20.0, 20.0, 22.0, 22.0), TargetClass.VESSEL, overall_conf=0.75)
    predictions = [pred_1, pred_2, pred_3]

    report = evaluator.evaluate(ground_truth, predictions, evaluated_area_km2=500.0)

    assert isinstance(report, BenchmarkReport)
    assert report.num_ground_truth == 3
    assert report.num_predictions == 3
    assert report.true_positives == 2
    assert report.false_positives == 1
    assert report.false_negatives == 1

    expected_precision = 2.0 / 3.0
    expected_recall = 2.0 / 3.0
    expected_f1 = 2.0 / 3.0

    assert abs(report.precision - expected_precision) < 1e-4
    assert abs(report.recall - expected_recall) < 1e-4
    assert abs(report.f1_score - expected_f1) < 1e-4
    assert report.average_precision > 0.0

    assert "VESSEL" in report.class_metrics
    assert "AIRCRAFT" in report.class_metrics
    assert report.class_metrics["VESSEL"]["tp"] == 1.0
    assert report.class_metrics["VESSEL"]["fn"] == 1.0
    assert report.class_metrics["VESSEL"]["fp"] == 1.0
    assert report.class_metrics["AIRCRAFT"]["tp"] == 1.0
    assert report.class_metrics["AIRCRAFT"]["fn"] == 0.0
    assert report.class_metrics["AIRCRAFT"]["fp"] == 0.0

    out_file = tmp_path / "report.json"
    report.save_json(out_file)
    assert out_file.exists()

    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["true_positives"] == 2
    assert loaded["pipeline_version"] == "2.0.0"
    assert loaded["git_commit"] == "30d2bc8"


def test_false_alarm_rate_per_km2() -> None:
    evaluator = BenchmarkEvaluator(iou_threshold=0.5)

    ground_truth: list[tuple[float, float, float, float]] = []
    predictions = [
        (0.0, 0.0, 1.0, 1.0),
        (2.0, 2.0, 3.0, 3.0),
        (4.0, 4.0, 5.0, 5.0),
        (6.0, 6.0, 7.0, 7.0),
    ]

    report = evaluator.evaluate(
        ground_truth,
        predictions,
        evaluated_area_km2=2000.0,
        total_eval_samples=100,
    )

    assert report.num_ground_truth == 0
    assert report.num_predictions == 4
    assert report.true_positives == 0
    assert report.false_positives == 4
    assert report.false_negatives == 0
    assert report.false_alarms_per_km2 == 4.0 / 2000.0
    assert report.probability_of_false_alarm == 4.0 / 100.0
