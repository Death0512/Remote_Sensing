from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import numpy as np
import pytest
from shapely.geometry import box

from src.benchmark import (
    BenchmarkReport,
    BenchmarkRunner,
    DatasetSplit,
    DetectorBenchmarkResult,
    GoldenDataset,
    GoldenSample,
    GroundTruthTarget,
    LeakageError,
    RegressionThresholds,
    SpatialLeakageError,
    TemporalLeakageError,
    calculate_average_precision,
    calculate_iou,
    compute_metrics,
    create_bitemporal_change_golden_dataset,
    create_optical_aircraft_golden_dataset,
    create_sar_vessel_golden_dataset,
    validate_no_leakage,
)
from src.domain.enums import SensorType, TargetClass


def _make_dummy_sample(
    sample_id: str,
    bounds: tuple[float, float, float, float],
    time_utc: datetime,
    split: DatasetSplit = DatasetSplit.TRAIN,
    area_km2: float = 1.0,
) -> GoldenSample:
    tgt = GroundTruthTarget(
        target_id=f"tgt-{sample_id}",
        target_class=TargetClass.VESSEL,
        geom_wgs84=box(bounds[0], bounds[1], bounds[2], bounds[3]),
        geom_px=(10.0, 10.0, 20.0, 20.0),
    )
    return GoldenSample(
        sample_id=sample_id,
        dataset_name="dummy-dataset",
        source="Test Source",
        license="MIT",
        acquisition_time_utc=time_utc,
        sensor=SensorType.S1_SAR,
        crs="EPSG:4326",
        gsd_m=10.0,
        bounds_wgs84=box(bounds[0], bounds[1], bounds[2], bounds[3]),
        ground_truth=(tgt,),
        checksum="dummy-hash",
        split=split,
        area_km2=area_km2,
    )


def test_split_no_leakage() -> None:
    t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

    s_train = _make_dummy_sample("s-tr-1", (100.0, 10.0, 101.0, 11.0), t0, DatasetSplit.TRAIN)
    s_test_clean = _make_dummy_sample("s-te-1", (102.0, 12.0, 103.0, 13.0), t1, DatasetSplit.TEST)

    assert validate_no_leakage([s_train], [s_test_clean], guard_window_seconds=86400.0) is True

    s_test_spatial_leak = _make_dummy_sample("s-te-leak-space", (100.5, 10.5, 101.5, 11.5), t1, DatasetSplit.TEST)
    with pytest.raises(SpatialLeakageError):
        validate_no_leakage([s_train], [s_test_spatial_leak], guard_window_seconds=0.0)

    t_close = t0 + timedelta(hours=2)
    s_test_temporal_leak = _make_dummy_sample("s-te-leak-time", (105.0, 15.0, 106.0, 16.0), t_close, DatasetSplit.TEST)
    with pytest.raises(TemporalLeakageError):
        validate_no_leakage([s_train], [s_test_temporal_leak], guard_window_seconds=86400.0)

    dataset = GoldenDataset(
        name="test-ds",
        train_samples=(s_train,),
        validation_samples=(s_test_clean,),
        test_samples=(s_test_clean,),
        guard_window_seconds=3600.0,
    )
    assert dataset.get_split(DatasetSplit.TRAIN) == (s_train,)
    assert len(dataset.all_samples()) == 3


def test_iou_and_ap_calculation() -> None:
    b1 = box(0.0, 0.0, 10.0, 10.0)
    b2 = box(0.0, 0.0, 10.0, 10.0)
    assert calculate_iou(b1, b2) == 1.0

    b3 = box(15.0, 15.0, 25.0, 25.0)
    assert calculate_iou(b1, b3) == 0.0

    b4 = box(5.0, 0.0, 15.0, 10.0)
    assert math.isclose(calculate_iou(b1, b4), 50.0 / 150.0, rel_tol=1e-5)

    gt_targets = [
        {"geom_wgs84": box(0.0, 0.0, 10.0, 10.0), "score": 1.0},
        {"geom_wgs84": box(20.0, 20.0, 30.0, 30.0), "score": 1.0},
    ]

    predictions_perfect = [
        {"geom_wgs84": box(0.0, 0.0, 10.0, 10.0), "score": 0.95},
        {"geom_wgs84": box(20.0, 20.0, 30.0, 30.0), "score": 0.85},
    ]
    ap_perfect = calculate_average_precision(gt_targets, predictions_perfect, iou_threshold=0.5)
    assert ap_perfect == 1.0

    predictions_mixed = [
        {"geom_wgs84": box(0.0, 0.0, 10.0, 10.0), "score": 0.9},
        {"geom_wgs84": box(100.0, 100.0, 110.0, 110.0), "score": 0.8},
        {"geom_wgs84": box(20.0, 20.0, 30.0, 30.0), "score": 0.7},
    ]
    ap_mixed = calculate_average_precision(gt_targets, predictions_mixed, iou_threshold=0.5)
    assert 0.0 < ap_mixed <= 1.0


def test_false_alarm_density_and_pfa_accuracy() -> None:
    gt_targets = [
        {"geom_wgs84": box(0.0, 0.0, 10.0, 10.0), "score": 1.0},
    ]
    predictions = [
        {"geom_wgs84": box(0.0, 0.0, 10.0, 10.0), "score": 0.9},
        {"geom_wgs84": box(50.0, 50.0, 60.0, 60.0), "score": 0.7},
        {"geom_wgs84": box(70.0, 70.0, 80.0, 80.0), "score": 0.6},
    ]

    area_km2 = 25.0
    metrics = compute_metrics(
        ground_truth=gt_targets,
        predictions=predictions,
        evaluated_area_km2=area_km2,
        total_eval_samples=100,
    )

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 2
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 1.0 / 3.0
    assert metrics["recall"] == 1.0
    assert math.isclose(metrics["false_alarms_per_km2"], 2.0 / 25.0)
    assert math.isclose(metrics["pfa"], 2.0 / 100.0)


def test_localization_rmse_accuracy() -> None:
    gt_targets = [
        {"geom_px": (10.0, 10.0, 20.0, 20.0), "score": 1.0},
        {"geom_px": (50.0, 50.0, 60.0, 60.0), "score": 1.0},
    ]
    predictions = [
        {"geom_px": (11.0, 12.0, 21.0, 22.0), "score": 0.9},
        {"geom_px": (50.0, 50.0, 60.0, 60.0), "score": 0.8},
    ]

    metrics = compute_metrics(
        ground_truth=gt_targets,
        predictions=predictions,
        evaluated_area_km2=1.0,
        gsd_m=2.0,
        prefer_px=True,
    )

    expected_d1 = math.sqrt(1.0**2 + 2.0**2)
    expected_d2 = 0.0
    expected_rmse_px = math.sqrt((expected_d1**2 + expected_d2**2) / 2.0)
    expected_rmse_m = expected_rmse_px * 2.0

    assert math.isclose(metrics["localization_rmse_px"], expected_rmse_px, rel_tol=1e-4)
    assert math.isclose(metrics["localization_rmse_m"], expected_rmse_m, rel_tol=1e-4)


def test_benchmark_report_complete() -> None:
    runner = BenchmarkRunner()
    report = runner.run_all(split=DatasetSplit.TEST)

    assert isinstance(report, BenchmarkReport)
    assert len(report.results) == 3
    assert report.passed is True
    assert report.pipeline_version == "3.0.0"
    assert report.summary["all_passed"] is True
    assert report.summary["total_detectors_evaluated"] == 3

    detector_names = {r.detector_name for r in report.results}
    assert "SarVesselDetector" in detector_names
    assert "OpticalAircraftDetector" in detector_names
    assert "BitemporalChangeDetector" in detector_names

    for res in report.results:
        assert isinstance(res, DetectorBenchmarkResult)
        assert res.split == DatasetSplit.TEST
        assert res.num_samples >= 1
        assert res.num_ground_truth >= 1
        assert res.true_positives >= 1
        assert res.pd >= 0.5
        assert res.recall >= 0.5
        assert res.precision >= 0.5
        assert res.f1_score >= 0.5
        assert res.ap50 >= 0.4
        assert res.mean_iou >= 0.5
        assert res.evaluated_area_km2 > 0.0
        assert res.latency_ms >= 0.0
        assert res.throughput_km2_per_s >= 0.0
        assert res.passed_thresholds is True
        assert len(res.threshold_failures) == 0


def test_per_detector_regression_thresholds_enforced() -> None:
    strict_thresholds = {
        "SarVesselDetector": RegressionThresholds(min_pd=0.999, min_precision=0.999),
        "OpticalAircraftDetector": RegressionThresholds(max_latency_ms=0.0001),
        "BitemporalChangeDetector": RegressionThresholds(min_ap50=0.999),
    }

    runner = BenchmarkRunner(thresholds=strict_thresholds)
    opt_res = runner.run_optical_aircraft_benchmark(split=DatasetSplit.TEST)

    assert opt_res.passed_thresholds is False
    assert len(opt_res.threshold_failures) >= 1
    assert any("latency_ms" in f for f in opt_res.threshold_failures)

    report = runner.run_all(split=DatasetSplit.TEST)
    assert report.passed is False
    assert report.summary["all_passed"] is False


def test_json_export_and_roundtrip(tmp_path: Path) -> None:
    runner = BenchmarkRunner()
    report = runner.run_all(split=DatasetSplit.TEST)

    out_file = tmp_path / "benchmark_report.json"
    report.save_json(out_file)
    assert out_file.exists()

    raw_json = out_file.read_text(encoding="utf-8")
    parsed_dict = json.loads(raw_json)
    assert parsed_dict["pipeline_version"] == "3.0.0"
    assert len(parsed_dict["results"]) == 3

    restored = BenchmarkReport.from_json(raw_json)
    assert restored.report_id == report.report_id
    assert restored.pipeline_version == report.pipeline_version
    assert restored.git_commit == report.git_commit
    assert restored.passed == report.passed
    assert len(restored.results) == len(report.results)

    for orig_r, rest_r in zip(report.results, restored.results):
        assert orig_r.detector_name == rest_r.detector_name
        assert orig_r.target_class == rest_r.target_class
        assert orig_r.split == rest_r.split
        assert orig_r.num_samples == rest_r.num_samples
        assert orig_r.num_ground_truth == rest_r.num_ground_truth
        assert orig_r.true_positives == rest_r.true_positives
        assert orig_r.precision == rest_r.precision
        assert orig_r.recall == rest_r.recall
        assert orig_r.f1_score == rest_r.f1_score
        assert orig_r.ap50 == rest_r.ap50
