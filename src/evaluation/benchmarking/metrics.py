from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional, Sequence, Union
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from src.domain.enums import Capability


def _calculate_iou(geom1: BaseGeometry, geom2: BaseGeometry) -> float:
    if not geom1.intersects(geom2):
        return 0.0
    inter_area = geom1.intersection(geom2).area
    union_area = geom1.union(geom2).area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


@dataclass(frozen=True)
class BenchmarkReport:
    pipeline_version: str
    git_commit: str
    execution_timestamp_utc: str
    num_ground_truth: int
    num_predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    average_precision: float
    probability_of_false_alarm: float
    false_alarms_per_km2: float
    evaluated_area_km2: float
    iou_threshold: float
    class_metrics: dict[str, dict[str, float]]
    matches: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, output_path: Union[str, Path]) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")


class BenchmarkEvaluator:
    capability: Capability = Capability.IMPLEMENTED

    def __init__(
        self,
        iou_threshold: float = 0.5,
        pipeline_version: str = "2.0.0",
        git_commit: str = "HEAD",
    ) -> None:
        if iou_threshold < 0.0 or iou_threshold > 1.0:
            raise ValueError("iou_threshold must be between 0.0 and 1.0")
        self.iou_threshold = iou_threshold
        self.pipeline_version = pipeline_version
        self.git_commit = git_commit

    def _extract_geometry(
        self,
        item: Union[BaseGeometry, tuple[float, float, float, float], list[float], dict[str, Any], Any],
    ) -> BaseGeometry:
        if isinstance(item, BaseGeometry):
            return item
        if isinstance(item, (tuple, list)):
            if len(item) == 4:
                return box(item[0], item[1], item[2], item[3])
            raise ValueError("Tuple/list bbox must have 4 elements: (minx, miny, maxx, maxy)")
        if isinstance(item, dict):
            if "geom" in item:
                return self._extract_geometry(item["geom"])
            if "bbox" in item:
                return self._extract_geometry(item["bbox"])
            if "geom_wgs84" in item:
                return self._extract_geometry(item["geom_wgs84"])
        if hasattr(item, "geom_wgs84"):
            return getattr(item, "geom_wgs84")
        if hasattr(item, "geom"):
            return getattr(item, "geom")
        if hasattr(item, "bbox"):
            return self._extract_geometry(getattr(item, "bbox"))
        raise TypeError(f"Cannot extract geometry from object of type {type(item)}")

    def _extract_class(self, item: Any) -> str:
        if isinstance(item, dict):
            val = item.get("target_class") or item.get("class") or item.get("label") or "DEFAULT"
            return getattr(val, "value", str(val))
        if hasattr(item, "target_class"):
            val = getattr(item, "target_class")
            return getattr(val, "value", str(val))
        if hasattr(item, "label"):
            val = getattr(item, "label")
            return getattr(val, "value", str(val))
        return "DEFAULT"

    def _extract_score(self, item: Any) -> float:
        if isinstance(item, dict):
            if "score" in item:
                return float(item["score"])
            if "confidence" in item:
                conf = item["confidence"]
                if hasattr(conf, "raw_score"):
                    return float(conf.raw_score)
                if hasattr(conf, "overall"):
                    return float(conf.overall)
                if isinstance(conf, (int, float)):
                    return float(conf)
            return 1.0
        if hasattr(item, "confidence"):
            conf = getattr(item, "confidence")
            if hasattr(conf, "raw_score"):
                return float(conf.raw_score)
            if hasattr(conf, "overall"):
                return float(conf.overall)
            if isinstance(conf, (int, float)):
                return float(conf)
        if hasattr(item, "score"):
            return float(getattr(item, "score"))
        return 1.0

    def compute_iou(self, geom1: BaseGeometry, geom2: BaseGeometry) -> float:
        return _calculate_iou(geom1, geom2)

    def match_detections(
        self,
        ground_truth: Sequence[Any],
        predictions: Sequence[Any],
        iou_threshold: Optional[float] = None,
    ) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
        thresh = self.iou_threshold if iou_threshold is None else iou_threshold
        num_gt = len(ground_truth)
        num_pred = len(predictions)

        if num_gt == 0:
            return [], [], list(range(num_pred))
        if num_pred == 0:
            return [], list(range(num_gt)), []

        gt_geoms = [self._extract_geometry(gt) for gt in ground_truth]
        pred_geoms = [self._extract_geometry(p) for p in predictions]

        iou_matrix = np.zeros((num_gt, num_pred), dtype=np.float64)
        for i, g in enumerate(gt_geoms):
            for j, p in enumerate(pred_geoms):
                iou_matrix[i, j] = _calculate_iou(g, p)

        cost_matrix = -iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched: list[tuple[int, int, float]] = []
        unmatched_gt = set(range(num_gt))
        unmatched_pred = set(range(num_pred))

        for r, c in zip(row_ind, col_ind):
            iou_val = float(iou_matrix[r, c])
            if iou_val >= thresh:
                matched.append((int(r), int(c), iou_val))
                unmatched_gt.discard(r)
                unmatched_pred.discard(c)

        return matched, sorted(list(unmatched_gt)), sorted(list(unmatched_pred))

    def evaluate(
        self,
        ground_truth: Sequence[Any],
        predictions: Sequence[Any],
        evaluated_area_km2: float = 0.0,
        total_eval_samples: Optional[int] = None,
    ) -> BenchmarkReport:
        if evaluated_area_km2 < 0.0:
            raise ValueError("evaluated_area_km2 cannot be negative")

        matched, unmatched_gt, unmatched_pred = self.match_detections(
            ground_truth, predictions, self.iou_threshold
        )

        tp = len(matched)
        fn = len(unmatched_gt)
        fp = len(unmatched_pred)

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (
            float(2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        n_samples = total_eval_samples if total_eval_samples is not None else (tp + fp + fn)
        p_fa = float(fp / n_samples) if n_samples > 0 else 0.0

        false_alarms_per_km2 = (
            float(fp / evaluated_area_km2) if evaluated_area_km2 > 0.0 else 0.0
        )

        ap = self._calculate_average_precision(ground_truth, predictions)
        class_metrics = self._calculate_class_metrics(
            ground_truth, predictions, matched, unmatched_gt, unmatched_pred
        )

        match_records = [
            {"gt_idx": r, "pred_idx": c, "iou": iou} for r, c, iou in matched
        ]

        return BenchmarkReport(
            pipeline_version=self.pipeline_version,
            git_commit=self.git_commit,
            execution_timestamp_utc=datetime.now(timezone.utc).isoformat(),
            num_ground_truth=len(ground_truth),
            num_predictions=len(predictions),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            average_precision=ap,
            probability_of_false_alarm=p_fa,
            false_alarms_per_km2=false_alarms_per_km2,
            evaluated_area_km2=evaluated_area_km2,
            iou_threshold=self.iou_threshold,
            class_metrics=class_metrics,
            matches=match_records,
        )

    def _calculate_average_precision(
        self, ground_truth: Sequence[Any], predictions: Sequence[Any]
    ) -> float:
        num_gt = len(ground_truth)
        if num_gt == 0:
            return 0.0
        if len(predictions) == 0:
            return 0.0

        indexed_preds = list(enumerate(predictions))
        indexed_preds.sort(
            key=lambda item: self._extract_score(item[1]), reverse=True
        )

        gt_geoms = [self._extract_geometry(gt) for gt in ground_truth]
        matched_gt: set[int] = set()

        tp_flags = []
        fp_flags = []

        for _, pred in indexed_preds:
            p_geom = self._extract_geometry(pred)
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, g_geom in enumerate(gt_geoms):
                iou = _calculate_iou(g_geom, p_geom)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= self.iou_threshold and best_gt_idx not in matched_gt:
                tp_flags.append(1)
                fp_flags.append(0)
                matched_gt.add(best_gt_idx)
            else:
                tp_flags.append(0)
                fp_flags.append(1)

        cum_tp = np.cumsum(tp_flags)
        cum_fp = np.cumsum(fp_flags)

        precisions = cum_tp / (cum_tp + cum_fp)
        recalls = cum_tp / num_gt

        precisions = np.concatenate(([0.0], precisions, [0.0]))
        recalls = np.concatenate(([0.0], recalls, [1.0]))

        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        return float(ap)

    def _calculate_class_metrics(
        self,
        ground_truth: Sequence[Any],
        predictions: Sequence[Any],
        matched: list[tuple[int, int, float]],
        unmatched_gt: list[int],
        unmatched_pred: list[int],
    ) -> dict[str, dict[str, float]]:
        all_classes: set[str] = set()
        gt_classes = [self._extract_class(gt) for gt in ground_truth]
        pred_classes = [self._extract_class(p) for p in predictions]

        all_classes.update(gt_classes)
        all_classes.update(pred_classes)

        metrics: dict[str, dict[str, float]] = {}

        for cls in sorted(list(all_classes)):
            c_tp = sum(
                1
                for r, c, _ in matched
                if gt_classes[r] == cls and pred_classes[c] == cls
            )
            c_fp = sum(
                1 for c in unmatched_pred if pred_classes[c] == cls
            ) + sum(
                1
                for r, c, _ in matched
                if pred_classes[c] == cls and gt_classes[r] != cls
            )
            c_fn = sum(
                1 for r in unmatched_gt if gt_classes[r] == cls
            ) + sum(
                1
                for r, c, _ in matched
                if gt_classes[r] == cls and pred_classes[c] != cls
            )

            p = float(c_tp / (c_tp + c_fp)) if (c_tp + c_fp) > 0 else 0.0
            r = float(c_tp / (c_tp + c_fn)) if (c_tp + c_fn) > 0 else 0.0
            f = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

            metrics[cls] = {
                "tp": float(c_tp),
                "fp": float(c_fp),
                "fn": float(c_fn),
                "precision": p,
                "recall": r,
                "f1_score": f,
            }

        return metrics
