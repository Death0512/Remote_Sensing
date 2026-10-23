import math
from typing import Any, Optional, Sequence, Union
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry


def calculate_iou(
    geom1: Union[BaseGeometry, tuple[float, float, float, float], list[float], Any],
    geom2: Union[BaseGeometry, tuple[float, float, float, float], list[float], Any],
    prefer_px: bool = False,
) -> float:
    g1 = extract_geometry(geom1, prefer_px=prefer_px)
    g2 = extract_geometry(geom2, prefer_px=prefer_px)
    if not g1.intersects(g2):
        return 0.0
    inter_area = g1.intersection(g2).area
    union_area = g1.union(g2).area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def extract_geometry(
    item: Union[BaseGeometry, tuple[float, float, float, float], list[float], dict[str, Any], Any],
    prefer_px: bool = False,
) -> BaseGeometry:
    if prefer_px:
        if isinstance(item, dict) and item.get("geom_px") is not None:
            px = item["geom_px"]
            if isinstance(px, (tuple, list)) and len(px) == 4:
                return box(px[0], px[1], px[2], px[3])
        if hasattr(item, "geom_px") and getattr(item, "geom_px") is not None:
            px = getattr(item, "geom_px")
            if isinstance(px, (tuple, list)) and len(px) == 4:
                return box(px[0], px[1], px[2], px[3])

    if isinstance(item, BaseGeometry):
        return item
    if isinstance(item, (tuple, list)):
        if len(item) == 4:
            return box(item[0], item[1], item[2], item[3])
        raise ValueError(f"Geometry tuple/list must have 4 coordinates, got {len(item)}")
    if isinstance(item, dict):
        if "geom_wgs84" in item:
            g = extract_geometry(item["geom_wgs84"], prefer_px=False)
            if g.area > 0.0:
                return g
            if item.get("geom_px") is not None:
                px = item["geom_px"]
                if isinstance(px, (tuple, list)) and len(px) == 4:
                    return box(px[0], px[1], px[2], px[3])
            return g
        if "geom" in item:
            return extract_geometry(item["geom"], prefer_px=False)
        if "bbox" in item:
            return extract_geometry(item["bbox"], prefer_px=False)
        if "bounds" in item:
            return extract_geometry(item["bounds"], prefer_px=False)
        if item.get("geom_px") is not None:
            px = item["geom_px"]
            if isinstance(px, (tuple, list)) and len(px) == 4:
                return box(px[0], px[1], px[2], px[3])
    if hasattr(item, "geom_wgs84"):
        g = extract_geometry(getattr(item, "geom_wgs84"), prefer_px=False)
        if g.area > 0.0:
            return g
        if hasattr(item, "geom_px") and getattr(item, "geom_px") is not None:
            px = getattr(item, "geom_px")
            if isinstance(px, (tuple, list)) and len(px) == 4:
                return box(px[0], px[1], px[2], px[3])
        return g
    if hasattr(item, "geometry"):
        return extract_geometry(getattr(item, "geometry"), prefer_px=False)
    if hasattr(item, "geom"):
        return extract_geometry(getattr(item, "geom"), prefer_px=False)
    if hasattr(item, "bounds_wgs84"):
        return extract_geometry(getattr(item, "bounds_wgs84"), prefer_px=False)
    if hasattr(item, "bounds"):
        return extract_geometry(getattr(item, "bounds"), prefer_px=False)
    if hasattr(item, "geom_px") and getattr(item, "geom_px") is not None:
        px = getattr(item, "geom_px")
        if isinstance(px, (tuple, list)) and len(px) == 4:
            return box(px[0], px[1], px[2], px[3])
    raise TypeError(f"Cannot extract geometry from item of type {type(item)}")


def extract_score(item: Any) -> float:
    if isinstance(item, dict):
        if "score" in item:
            return float(item["score"])
        if "confidence" in item:
            conf = item["confidence"]
            if hasattr(conf, "raw_score"):
                return float(conf.raw_score)
            if hasattr(conf, "model_score"):
                return float(conf.model_score)
            if isinstance(conf, (int, float)):
                return float(conf)
            if isinstance(conf, dict) and "raw_score" in conf:
                return float(conf["raw_score"])
        return 1.0
    if hasattr(item, "confidence"):
        conf = getattr(item, "confidence")
        if hasattr(conf, "raw_score"):
            return float(conf.raw_score)
        if hasattr(conf, "model_score"):
            return float(conf.model_score)
        if isinstance(conf, (int, float)):
            return float(conf)
    if hasattr(item, "score"):
        return float(getattr(item, "score"))
    return 1.0


def extract_centroid_wgs84(item: Any) -> tuple[float, float]:
    geom = extract_geometry(item)
    centroid = geom.centroid
    return float(centroid.x), float(centroid.y)


def extract_centroid_px(item: Any) -> Optional[tuple[float, float]]:
    if isinstance(item, dict):
        if "geom_px" in item and item["geom_px"] is not None:
            px = item["geom_px"]
            return float((px[0] + px[2]) / 2.0), float((px[1] + px[3]) / 2.0)
    if hasattr(item, "geom_px") and getattr(item, "geom_px") is not None:
        px = getattr(item, "geom_px")
        if isinstance(px, (tuple, list)) and len(px) == 4:
            return float((px[0] + px[2]) / 2.0), float((px[1] + px[3]) / 2.0)
    if hasattr(item, "attributes"):
        attrs = getattr(item, "attributes")
        attr_dict = dict(attrs) if isinstance(attrs, (tuple, list)) else (attrs if isinstance(attrs, dict) else {})
        if "centroid_col" in attr_dict and "centroid_row" in attr_dict:
            return float(attr_dict["centroid_col"]), float(attr_dict["centroid_row"])
        if "pixel_geometry_col" in attr_dict and "pixel_geometry_row" in attr_dict:
            return float(attr_dict["pixel_geometry_col"]), float(attr_dict["pixel_geometry_row"])
    return None


def haversine_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return float(r * c)


def _should_prefer_px(ground_truth: Sequence[Any], predictions: Sequence[Any]) -> bool:
    items = list(ground_truth) + list(predictions)
    if not items:
        return False
    return all(
        (hasattr(x, "geom_px") and getattr(x, "geom_px") is not None)
        or (isinstance(x, dict) and x.get("geom_px") is not None)
        for x in items
    )


def match_detections(
    ground_truth: Sequence[Any],
    predictions: Sequence[Any],
    iou_threshold: float = 0.5,
    prefer_px: Optional[bool] = None,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    num_gt = len(ground_truth)
    num_pred = len(predictions)
    if num_gt == 0:
        return [], [], list(range(num_pred))
    if num_pred == 0:
        return [], list(range(num_gt)), []

    if prefer_px is None:
        prefer_px = _should_prefer_px(ground_truth, predictions)

    gt_geoms = [extract_geometry(gt, prefer_px=prefer_px) for gt in ground_truth]
    pred_geoms = [extract_geometry(p, prefer_px=prefer_px) for p in predictions]

    iou_matrix = np.zeros((num_gt, num_pred), dtype=np.float64)
    for i, g in enumerate(gt_geoms):
        for j, p in enumerate(pred_geoms):
            iou_matrix[i, j] = calculate_iou(g, p, prefer_px=prefer_px)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matched: list[tuple[int, int, float]] = []
    unmatched_gt = set(range(num_gt))
    unmatched_pred = set(range(num_pred))

    for r, c in zip(row_ind, col_ind):
        iou_val = float(iou_matrix[r, c])
        if iou_val >= iou_threshold:
            matched.append((int(r), int(c), iou_val))
            unmatched_gt.discard(r)
            unmatched_pred.discard(c)

    return matched, sorted(list(unmatched_gt)), sorted(list(unmatched_pred))


def calculate_average_precision(
    ground_truth: Sequence[Any],
    predictions: Sequence[Any],
    iou_threshold: float = 0.5,
    prefer_px: Optional[bool] = None,
) -> float:
    num_gt = len(ground_truth)
    if num_gt == 0 or len(predictions) == 0:
        return 0.0

    if prefer_px is None:
        prefer_px = _should_prefer_px(ground_truth, predictions)

    indexed_preds = list(enumerate(predictions))
    indexed_preds.sort(key=lambda item: extract_score(item[1]), reverse=True)

    gt_geoms = [extract_geometry(gt, prefer_px=prefer_px) for gt in ground_truth]
    matched_gt: set[int] = set()

    tp_flags: list[int] = []
    fp_flags: list[int] = []

    for _, pred in indexed_preds:
        p_geom = extract_geometry(pred, prefer_px=prefer_px)
        best_iou = 0.0
        best_gt_idx = -1
        for g_idx, g_geom in enumerate(gt_geoms):
            iou = calculate_iou(g_geom, p_geom, prefer_px=prefer_px)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
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


def compute_metrics(
    ground_truth: Sequence[Any],
    predictions: Sequence[Any],
    evaluated_area_km2: float = 0.0,
    latency_ms: float = 0.0,
    memory_peak_mb: float = 0.0,
    gsd_m: float = 10.0,
    iou_threshold: float = 0.5,
    total_eval_samples: Optional[int] = None,
    prefer_px: Optional[bool] = None,
) -> dict[str, Any]:
    matched, unmatched_gt, unmatched_pred = match_detections(
        ground_truth, predictions, iou_threshold=iou_threshold, prefer_px=prefer_px
    )

    tp = len(matched)
    fn = len(unmatched_gt)
    fp = len(unmatched_pred)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    pd = recall
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    n_samples = total_eval_samples if total_eval_samples is not None else (tp + fp + fn)
    pfa = float(fp / n_samples) if n_samples > 0 else 0.0

    far_km2 = float(fp / evaluated_area_km2) if evaluated_area_km2 > 0.0 else 0.0

    ap50 = calculate_average_precision(
        ground_truth, predictions, iou_threshold=iou_threshold, prefer_px=prefer_px
    )

    mean_iou = float(np.mean([m[2] for m in matched])) if matched else 0.0

    px_errors: list[float] = []
    m_errors: list[float] = []

    for gt_idx, pred_idx, _ in matched:
        gt_item = ground_truth[gt_idx]
        pred_item = predictions[pred_idx]

        gt_px = extract_centroid_px(gt_item)
        pred_px = extract_centroid_px(pred_item)
        if gt_px is not None and pred_px is not None:
            d_px = math.sqrt((gt_px[0] - pred_px[0]) ** 2 + (gt_px[1] - pred_px[1]) ** 2)
            px_errors.append(d_px)
            m_errors.append(d_px * gsd_m)
        else:
            gt_lon, gt_lat = extract_centroid_wgs84(gt_item)
            pred_lon, pred_lat = extract_centroid_wgs84(pred_item)
            d_m = haversine_distance_m(gt_lon, gt_lat, pred_lon, pred_lat)
            m_errors.append(d_m)
            px_errors.append(d_m / gsd_m if gsd_m > 0 else 0.0)

    rmse_px = float(math.sqrt(float(np.mean(np.array(px_errors) ** 2)))) if px_errors else 0.0
    rmse_m = float(math.sqrt(float(np.mean(np.array(m_errors) ** 2)))) if m_errors else 0.0

    elapsed_s = latency_ms / 1000.0
    throughput = float(evaluated_area_km2 / elapsed_s) if elapsed_s > 0.0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "pd": pd,
        "recall": recall,
        "pfa": pfa,
        "precision": precision,
        "f1_score": f1,
        "ap50": ap50,
        "false_alarms_per_km2": far_km2,
        "localization_rmse_m": rmse_m,
        "localization_rmse_px": rmse_px,
        "mean_iou": mean_iou,
        "latency_ms": latency_ms,
        "memory_peak_mb": memory_peak_mb,
        "throughput_km2_per_s": throughput,
        "evaluated_area_km2": evaluated_area_km2,
        "matched": matched,
    }
