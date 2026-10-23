from datetime import datetime, timezone
import time
import tracemalloc
from typing import Any, Optional, Sequence, Union
from uuid import uuid4

from src.benchmark.contracts import (
    BenchmarkReport,
    DatasetSplit,
    DetectorBenchmarkResult,
    GoldenSample,
    RegressionThresholds,
)
from src.benchmark.datasets import (
    GoldenDataset,
    create_bitemporal_change_golden_dataset,
    create_optical_aircraft_golden_dataset,
    create_sar_vessel_golden_dataset,
)
from src.benchmark.metrics import compute_metrics
from src.detection.change.change_detector import BitemporalChangeDetector
from src.detection.optical.aircraft_detector import OpticalAircraftDetector
from src.detection.sar.local_cfar import LocalCfarDetector
from src.detection.sar.vessel_detector import SarVesselDetector
from src.domain.enums import TargetClass


class BenchmarkRunner:
    def __init__(
        self,
        thresholds: Optional[dict[str, RegressionThresholds]] = None,
        pipeline_version: str = "3.0.0",
        git_commit: str = "3536f55",
    ) -> None:
        self.pipeline_version = pipeline_version
        self.git_commit = git_commit
        default_th = RegressionThresholds()
        self.thresholds = thresholds or {
            "SarVesselDetector": default_th,
            "OpticalAircraftDetector": default_th,
            "BitemporalChangeDetector": default_th,
        }

    def _get_thresholds(self, name: str) -> RegressionThresholds:
        if name in self.thresholds:
            return self.thresholds[name]
        return RegressionThresholds()

    def _evaluate_thresholds(
        self,
        name: str,
        metrics: dict[str, Any],
    ) -> tuple[bool, tuple[str, ...]]:
        th = self._get_thresholds(name)
        failures: list[str] = []

        if metrics["pd"] < th.min_pd:
            failures.append(f"pd {metrics['pd']:.3f} < min {th.min_pd:.3f}")
        if metrics["precision"] < th.min_precision:
            failures.append(f"precision {metrics['precision']:.3f} < min {th.min_precision:.3f}")
        if metrics["f1_score"] < th.min_f1:
            failures.append(f"f1_score {metrics['f1_score']:.3f} < min {th.min_f1:.3f}")
        if metrics["ap50"] < th.min_ap50:
            failures.append(f"ap50 {metrics['ap50']:.3f} < min {th.min_ap50:.3f}")
        if metrics["pfa"] > th.max_pfa:
            failures.append(f"pfa {metrics['pfa']:.4f} > max {th.max_pfa:.4f}")
        if metrics["false_alarms_per_km2"] > th.max_far_per_km2:
            failures.append(f"far_km2 {metrics['false_alarms_per_km2']:.2f} > max {th.max_far_per_km2:.2f}")
        if metrics["localization_rmse_px"] > th.max_localization_rmse_px:
            failures.append(f"rmse_px {metrics['localization_rmse_px']:.2f} > max {th.max_localization_rmse_px:.2f}")
        if metrics["latency_ms"] > th.max_latency_ms:
            failures.append(f"latency_ms {metrics['latency_ms']:.2f} > max {th.max_latency_ms:.2f}")

        return len(failures) == 0, tuple(failures)

    def run_sar_vessel_benchmark(
        self,
        dataset: Optional[GoldenDataset] = None,
        detector: Optional[SarVesselDetector] = None,
        split: DatasetSplit = DatasetSplit.TEST,
    ) -> DetectorBenchmarkResult:
        ds = dataset or create_sar_vessel_golden_dataset()
        samples = ds.get_split(split)

        if detector is None:
            cfar = LocalCfarDetector(guard_radius_px=3, background_radius_px=10, pfa=1e-5)
            detector = SarVesselDetector(
                cfar_detector=cfar,
                min_length_m=10.0,
                max_length_m=450.0,
                min_width_m=2.0,
                max_width_m=100.0,
                min_aspect_ratio=1.0,
                max_aspect_ratio=15.0,
                pixel_spacing_m=10.0,
            )

        all_gt: list[Any] = []
        all_preds: list[Any] = []
        total_area_km2 = 0.0

        tracemalloc.start()
        start_time = time.perf_counter()

        for s in samples:
            total_area_km2 += s.compute_area_km2()
            all_gt.extend(s.ground_truth)
            preds = detector.detect(
                intensity=s.image_data,
                land_mask=s.land_mask,
                pixel_spacing_m=s.gsd_m,
                acquisition_time_utc=s.acquisition_time_utc,
            )
            if hasattr(preds, "__iter__"):
                all_preds.extend(list(preds))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_mem / (1024.0 * 1024.0)

        m = compute_metrics(
            ground_truth=all_gt,
            predictions=all_preds,
            evaluated_area_km2=total_area_km2,
            latency_ms=elapsed_ms,
            memory_peak_mb=peak_mb,
            gsd_m=samples[0].gsd_m if samples else 10.0,
        )

        passed, failures = self._evaluate_thresholds("SarVesselDetector", m)

        return DetectorBenchmarkResult(
            detector_name="SarVesselDetector",
            target_class=TargetClass.VESSEL,
            split=split,
            num_samples=len(samples),
            num_ground_truth=len(all_gt),
            num_predictions=len(all_preds),
            true_positives=m["true_positives"],
            false_positives=m["false_positives"],
            false_negatives=m["false_negatives"],
            pd=m["pd"],
            recall=m["recall"],
            pfa=m["pfa"],
            precision=m["precision"],
            f1_score=m["f1_score"],
            ap50=m["ap50"],
            false_alarms_per_km2=m["false_alarms_per_km2"],
            localization_rmse_m=m["localization_rmse_m"],
            localization_rmse_px=m["localization_rmse_px"],
            mean_iou=m["mean_iou"],
            latency_ms=m["latency_ms"],
            memory_peak_mb=peak_mb,
            throughput_km2_per_s=m["throughput_km2_per_s"],
            evaluated_area_km2=total_area_km2,
            passed_thresholds=passed,
            threshold_failures=failures,
        )

    def run_optical_aircraft_benchmark(
        self,
        dataset: Optional[GoldenDataset] = None,
        detector: Optional[OpticalAircraftDetector] = None,
        split: DatasetSplit = DatasetSplit.TEST,
    ) -> DetectorBenchmarkResult:
        ds = dataset or create_optical_aircraft_golden_dataset()
        samples = ds.get_split(split)

        if detector is None:
            detector = OpticalAircraftDetector(
                weights_path=None,
                min_length_m=5.0,
                max_length_m=80.0,
                pixel_spacing_m=1.0,
            )

        all_gt: list[Any] = []
        all_preds: list[Any] = []
        total_area_km2 = 0.0

        tracemalloc.start()
        start_time = time.perf_counter()

        for s in samples:
            total_area_km2 += s.compute_area_km2()
            all_gt.extend(s.ground_truth)
            if s.processed_observation is not None:
                preds = detector.detect(
                    processed_observation=s.processed_observation,
                    image=s.image_data,
                    pixel_spacing_m=s.gsd_m,
                    acquisition_time_utc=s.acquisition_time_utc,
                )
                if hasattr(preds, "__iter__"):
                    all_preds.extend(list(preds))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_mem / (1024.0 * 1024.0)

        m = compute_metrics(
            ground_truth=all_gt,
            predictions=all_preds,
            evaluated_area_km2=total_area_km2,
            latency_ms=elapsed_ms,
            memory_peak_mb=peak_mb,
            gsd_m=samples[0].gsd_m if samples else 1.0,
        )

        passed, failures = self._evaluate_thresholds("OpticalAircraftDetector", m)

        return DetectorBenchmarkResult(
            detector_name="OpticalAircraftDetector",
            target_class=TargetClass.AIRCRAFT,
            split=split,
            num_samples=len(samples),
            num_ground_truth=len(all_gt),
            num_predictions=len(all_preds),
            true_positives=m["true_positives"],
            false_positives=m["false_positives"],
            false_negatives=m["false_negatives"],
            pd=m["pd"],
            recall=m["recall"],
            pfa=m["pfa"],
            precision=m["precision"],
            f1_score=m["f1_score"],
            ap50=m["ap50"],
            false_alarms_per_km2=m["false_alarms_per_km2"],
            localization_rmse_m=m["localization_rmse_m"],
            localization_rmse_px=m["localization_rmse_px"],
            mean_iou=m["mean_iou"],
            latency_ms=m["latency_ms"],
            memory_peak_mb=peak_mb,
            throughput_km2_per_s=m["throughput_km2_per_s"],
            evaluated_area_km2=total_area_km2,
            passed_thresholds=passed,
            threshold_failures=failures,
        )

    def run_bitemporal_change_benchmark(
        self,
        dataset: Optional[GoldenDataset] = None,
        detector: Optional[BitemporalChangeDetector] = None,
        split: DatasetSplit = DatasetSplit.TEST,
    ) -> DetectorBenchmarkResult:
        ds = dataset or create_bitemporal_change_golden_dataset()
        samples = ds.get_split(split)

        if detector is None:
            detector = BitemporalChangeDetector(
                min_area_px=10,
                morph_kernel_size=1,
            )

        all_gt: list[Any] = []
        all_preds: list[Any] = []
        total_area_km2 = 0.0

        tracemalloc.start()
        start_time = time.perf_counter()

        for s in samples:
            total_area_km2 += s.compute_area_km2()
            all_gt.extend(s.ground_truth)
            preds = detector.detect(s.image_data)
            if hasattr(preds, "__iter__"):
                all_preds.extend(list(preds))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_mem / (1024.0 * 1024.0)

        m = compute_metrics(
            ground_truth=all_gt,
            predictions=all_preds,
            evaluated_area_km2=total_area_km2,
            latency_ms=elapsed_ms,
            memory_peak_mb=peak_mb,
            gsd_m=samples[0].gsd_m if samples else 10.0,
        )

        passed, failures = self._evaluate_thresholds("BitemporalChangeDetector", m)

        return DetectorBenchmarkResult(
            detector_name="BitemporalChangeDetector",
            target_class=TargetClass.INFRASTRUCTURE_CHANGE,
            split=split,
            num_samples=len(samples),
            num_ground_truth=len(all_gt),
            num_predictions=len(all_preds),
            true_positives=m["true_positives"],
            false_positives=m["false_positives"],
            false_negatives=m["false_negatives"],
            pd=m["pd"],
            recall=m["recall"],
            pfa=m["pfa"],
            precision=m["precision"],
            f1_score=m["f1_score"],
            ap50=m["ap50"],
            false_alarms_per_km2=m["false_alarms_per_km2"],
            localization_rmse_m=m["localization_rmse_m"],
            localization_rmse_px=m["localization_rmse_px"],
            mean_iou=m["mean_iou"],
            latency_ms=m["latency_ms"],
            memory_peak_mb=peak_mb,
            throughput_km2_per_s=m["throughput_km2_per_s"],
            evaluated_area_km2=total_area_km2,
            passed_thresholds=passed,
            threshold_failures=failures,
        )

    def run_all(
        self,
        golden_datasets: Optional[dict[str, GoldenDataset]] = None,
        split: DatasetSplit = DatasetSplit.TEST,
    ) -> BenchmarkReport:
        gds = golden_datasets or {}
        sar_ds = gds.get("sar", create_sar_vessel_golden_dataset())
        opt_ds = gds.get("optical", create_optical_aircraft_golden_dataset())
        chg_ds = gds.get("change", create_bitemporal_change_golden_dataset())

        sar_res = self.run_sar_vessel_benchmark(dataset=sar_ds, split=split)
        opt_res = self.run_optical_aircraft_benchmark(dataset=opt_ds, split=split)
        chg_res = self.run_bitemporal_change_benchmark(dataset=chg_ds, split=split)

        results = (sar_res, opt_res, chg_res)
        all_passed = all(r.passed_thresholds for r in results)

        summary = {
            "total_detectors_evaluated": len(results),
            "total_ground_truth": sum(r.num_ground_truth for r in results),
            "total_predictions": sum(r.num_predictions for r in results),
            "total_true_positives": sum(r.true_positives for r in results),
            "total_false_positives": sum(r.false_positives for r in results),
            "mean_f1_score": round(float(sum(r.f1_score for r in results) / len(results)), 4),
            "mean_ap50": round(float(sum(r.ap50 for r in results) / len(results)), 4),
            "total_evaluated_area_km2": round(float(sum(r.evaluated_area_km2 for r in results)), 4),
            "all_passed": all_passed,
        }

        return BenchmarkReport(
            report_id=f"rep-bench-{uuid4().hex[:12]}",
            pipeline_version=self.pipeline_version,
            git_commit=self.git_commit,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            results=results,
            passed=all_passed,
            summary=summary,
        )
