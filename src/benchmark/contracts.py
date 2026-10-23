from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import enum
import json
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry

from src.domain.enums import SensorType, TargetClass


class DatasetSplit(str, enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class GroundTruthTarget:
    target_id: str
    target_class: TargetClass
    geom_wgs84: Any
    geom_px: Optional[tuple[float, float, float, float]] = None
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        geom_val = self.geom_wgs84
        if isinstance(geom_val, BaseGeometry):
            geom_val = geom_val.__geo_interface__
        return {
            "target_id": self.target_id,
            "target_class": self.target_class.value if isinstance(self.target_class, TargetClass) else str(self.target_class),
            "geom_wgs84": geom_val,
            "geom_px": list(self.geom_px) if self.geom_px is not None else None,
            "properties": dict(self.properties),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GroundTruthTarget":
        raw_geom = data["geom_wgs84"]
        geom = shape(raw_geom) if isinstance(raw_geom, dict) else raw_geom
        target_class = TargetClass(data["target_class"]) if isinstance(data["target_class"], str) else data["target_class"]
        geom_px = tuple(data["geom_px"]) if data.get("geom_px") is not None else None
        return cls(
            target_id=str(data["target_id"]),
            target_class=target_class,
            geom_wgs84=geom,
            geom_px=geom_px,
            properties=dict(data.get("properties", {})),
        )


@dataclass(frozen=True)
class GoldenSample:
    sample_id: str
    dataset_name: str
    source: str
    license: str
    acquisition_time_utc: datetime
    sensor: SensorType
    crs: str
    gsd_m: float
    bounds_wgs84: Any
    ground_truth: tuple[GroundTruthTarget, ...]
    checksum: str
    split: DatasetSplit
    metadata: dict[str, Any] = field(default_factory=dict)
    image_data: Optional[Any] = None
    land_mask: Optional[Any] = None
    processed_observation: Optional[Any] = None
    area_km2: Optional[float] = None

    @property
    def resolution_m(self) -> float:
        return self.gsd_m

    def compute_area_km2(self) -> float:
        if self.area_km2 is not None and self.area_km2 > 0.0:
            return float(self.area_km2)
        if self.image_data is not None:
            if isinstance(self.image_data, tuple):
                arr = self.image_data[0]
            else:
                arr = self.image_data
            if hasattr(arr, "shape") and len(arr.shape) >= 2:
                h, w = arr.shape[-2], arr.shape[-1]
                area_m2 = (h * self.gsd_m) * (w * self.gsd_m)
                return float(area_m2 / 1_000_000.0)
        if isinstance(self.bounds_wgs84, (tuple, list)) and len(self.bounds_wgs84) == 4:
            minx, miny, maxx, maxy = self.bounds_wgs84
            geom = box(minx, miny, maxx, maxy)
        elif isinstance(self.bounds_wgs84, BaseGeometry):
            geom = self.bounds_wgs84
        else:
            return 1.0
        min_lon, min_lat, max_lon, max_lat = geom.bounds
        lat_mid = (min_lat + max_lat) / 2.0
        import math
        lat_dist_m = abs(max_lat - min_lat) * 111320.0
        lon_dist_m = abs(max_lon - min_lon) * 111320.0 * math.cos(math.radians(lat_mid))
        return float((lat_dist_m * lon_dist_m) / 1_000_000.0)


@dataclass(frozen=True)
class RegressionThresholds:
    min_pd: float = 0.5
    min_precision: float = 0.5
    min_f1: float = 0.5
    min_ap50: float = 0.4
    max_pfa: float = 0.2
    max_far_per_km2: float = 200.0
    max_localization_rmse_px: float = 20.0
    max_latency_ms: float = 10000.0


@dataclass(frozen=True)
class DetectorBenchmarkResult:
    detector_name: str
    target_class: TargetClass
    split: DatasetSplit
    num_samples: int
    num_ground_truth: int
    num_predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    pd: float
    recall: float
    pfa: float
    precision: float
    f1_score: float
    ap50: float
    false_alarms_per_km2: float
    localization_rmse_m: float
    localization_rmse_px: float
    mean_iou: float
    latency_ms: float
    memory_peak_mb: float
    throughput_km2_per_s: float
    evaluated_area_km2: float
    passed_thresholds: bool = True
    threshold_failures: tuple[str, ...] = ()
    condition_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_name": self.detector_name,
            "target_class": self.target_class.value if isinstance(self.target_class, TargetClass) else str(self.target_class),
            "split": self.split.value if isinstance(self.split, DatasetSplit) else str(self.split),
            "num_samples": self.num_samples,
            "num_ground_truth": self.num_ground_truth,
            "num_predictions": self.num_predictions,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "pd": round(float(self.pd), 4),
            "recall": round(float(self.recall), 4),
            "pfa": round(float(self.pfa), 4),
            "precision": round(float(self.precision), 4),
            "f1_score": round(float(self.f1_score), 4),
            "ap50": round(float(self.ap50), 4),
            "false_alarms_per_km2": round(float(self.false_alarms_per_km2), 4),
            "localization_rmse_m": round(float(self.localization_rmse_m), 4),
            "localization_rmse_px": round(float(self.localization_rmse_px), 4),
            "mean_iou": round(float(self.mean_iou), 4),
            "latency_ms": round(float(self.latency_ms), 2),
            "memory_peak_mb": round(float(self.memory_peak_mb), 2),
            "throughput_km2_per_s": round(float(self.throughput_km2_per_s), 4),
            "evaluated_area_km2": round(float(self.evaluated_area_km2), 4),
            "passed_thresholds": self.passed_thresholds,
            "threshold_failures": list(self.threshold_failures),
            "condition_metrics": dict(self.condition_metrics),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectorBenchmarkResult":
        target_class = TargetClass(data["target_class"]) if isinstance(data["target_class"], str) else data["target_class"]
        split = DatasetSplit(data["split"]) if isinstance(data["split"], str) else data["split"]
        return cls(
            detector_name=str(data["detector_name"]),
            target_class=target_class,
            split=split,
            num_samples=int(data["num_samples"]),
            num_ground_truth=int(data["num_ground_truth"]),
            num_predictions=int(data["num_predictions"]),
            true_positives=int(data["true_positives"]),
            false_positives=int(data["false_positives"]),
            false_negatives=int(data["false_negatives"]),
            pd=float(data["pd"]),
            recall=float(data["recall"]),
            pfa=float(data["pfa"]),
            precision=float(data["precision"]),
            f1_score=float(data["f1_score"]),
            ap50=float(data["ap50"]),
            false_alarms_per_km2=float(data["false_alarms_per_km2"]),
            localization_rmse_m=float(data["localization_rmse_m"]),
            localization_rmse_px=float(data["localization_rmse_px"]),
            mean_iou=float(data["mean_iou"]),
            latency_ms=float(data["latency_ms"]),
            memory_peak_mb=float(data.get("memory_peak_mb", 0.0)),
            throughput_km2_per_s=float(data["throughput_km2_per_s"]),
            evaluated_area_km2=float(data["evaluated_area_km2"]),
            passed_thresholds=bool(data.get("passed_thresholds", True)),
            threshold_failures=tuple(data.get("threshold_failures", ())),
            condition_metrics=dict(data.get("condition_metrics", {})),
        )


@dataclass(frozen=True)
class BenchmarkReport:
    report_id: str
    pipeline_version: str
    git_commit: str
    timestamp_utc: str
    results: tuple[DetectorBenchmarkResult, ...]
    passed: bool
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "pipeline_version": self.pipeline_version,
            "git_commit": self.git_commit,
            "timestamp_utc": self.timestamp_utc,
            "results": [r.to_dict() for r in self.results],
            "passed": self.passed,
            "summary": dict(self.summary),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkReport":
        results = tuple(DetectorBenchmarkResult.from_dict(r) for r in data["results"])
        return cls(
            report_id=str(data["report_id"]),
            pipeline_version=str(data["pipeline_version"]),
            git_commit=str(data["git_commit"]),
            timestamp_utc=str(data["timestamp_utc"]),
            results=results,
            passed=bool(data["passed"]),
            summary=dict(data.get("summary", {})),
        )

    @classmethod
    def from_json(cls, text: str) -> "BenchmarkReport":
        return cls.from_dict(json.loads(text))
