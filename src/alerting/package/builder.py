from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional, Union
import warnings

import numpy as np
from PIL import Image
import rasterio
import shapely.geometry

from src.alerting.priority import validate_priority_gating
from src.alerting.review.store import ReviewAuditRecord
from src.domain.alert import Alert
from src.domain.detection import Detection
from src.domain.enums import Capability
from src.domain.evidence import Evidence
from src.domain.hypothesis import Hypothesis
from src.domain.provenance import Provenance


def _serialize_provenance(prov: Optional[Provenance]) -> Optional[dict[str, Any]]:
    if prov is None:
        return None
    return {
        "provenance_id": prov.provenance_id,
        "source_asset_id": prov.source_asset_id,
        "acquisition_metadata_ref": prov.acquisition_metadata_ref,
        "pipeline_version": prov.pipeline_version,
        "git_commit": prov.git_commit,
        "config_hash": prov.config_hash,
        "model_id": prov.model_id,
        "model_version": prov.model_version,
        "input_checksum": prov.input_checksum,
        "output_checksum": prov.output_checksum,
        "processing_run_id": prov.processing_run_id,
        "processing_time_utc": prov.processing_time_utc.isoformat()
        if hasattr(prov.processing_time_utc, "isoformat")
        else str(prov.processing_time_utc),
        "parent_provenance_id": prov.parent_provenance_id,
    }


def _serialize_evidence(ev: Any) -> dict[str, Any]:
    if isinstance(ev, Evidence):
        return {
            "evidence_id": ev.evidence_id,
            "kind": ev.kind.value if hasattr(ev.kind, "value") else str(ev.kind),
            "value": ev.value,
            "quality": ev.quality,
            "source": ev.source,
            "timestamp_utc": ev.timestamp_utc.isoformat()
            if hasattr(ev.timestamp_utc, "isoformat")
            else str(ev.timestamp_utc),
            "uncertainty": ev.uncertainty,
            "supports": ev.supports,
            "references": list(ev.references),
            "provenance": _serialize_provenance(ev.provenance) if hasattr(ev, "provenance") and ev.provenance else None,
        }
    elif isinstance(ev, dict):
        return ev
    return {"raw": str(ev)}


def _write_raw_tiff(path: Path, arr: np.ndarray) -> None:
    if arr.ndim == 2:
        if arr.dtype in (np.float32, np.float64):
            Image.fromarray(arr.astype(np.float32)).save(path)
        elif arr.dtype in (np.uint16, np.int16, np.uint8):
            Image.fromarray(arr).save(path)
        else:
            Image.fromarray(arr.astype(np.float32)).save(path)
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1]:
            channels, h, w = arr.shape
            bands_data = arr
        elif arr.shape[2] in (1, 3, 4):
            h, w, channels = arr.shape
            bands_data = np.transpose(arr, (2, 0, 1))
        else:
            channels, h, w = arr.shape
            bands_data = arr
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(
                str(path),
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=channels,
                dtype=str(arr.dtype),
            ) as dst:
                dst.write(bands_data)
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")


def _write_preview_png(path: Path, arr: np.ndarray) -> None:
    if arr.dtype == np.uint8 and arr.ndim in (2, 3):
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1]:
            arr_img = np.transpose(arr, (1, 2, 0))
        else:
            arr_img = arr
        if arr_img.ndim == 3 and arr_img.shape[2] == 1:
            arr_img = arr_img[:, :, 0]
        Image.fromarray(arr_img).save(path)
        return

    float_arr = arr.astype(np.float32)
    if float_arr.ndim == 3 and float_arr.shape[0] in (1, 3, 4) and float_arr.shape[0] < float_arr.shape[1]:
        float_arr = np.transpose(float_arr, (1, 2, 0))
    if float_arr.ndim == 3 and float_arr.shape[2] == 1:
        float_arr = float_arr[:, :, 0]

    p2 = float(np.nanpercentile(float_arr, 2))
    p98 = float(np.nanpercentile(float_arr, 98))
    if p98 > p2:
        stretched = np.clip((float_arr - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
    else:
        stretched = np.clip(float_arr, 0, 255).astype(np.uint8)
    Image.fromarray(stretched).save(path)


def _write_analysis_tensor(path: Path, arr: np.ndarray) -> None:
    norm_arr = arr.astype(np.float32)
    np.save(path.with_suffix(".npy"), norm_arr)
    if norm_arr.ndim in (2, 3):
        try:
            _write_raw_tiff(path.with_suffix(".tif"), norm_arr)
        except Exception:
            pass


class AlertPackageBuilder:
    capability: Capability = Capability.IMPLEMENTED

    def build_package(
        self,
        alert: Alert,
        hypothesis: Hypothesis,
        chips: dict[str, Any],
        output_dir: Union[str, Path],
        detection: Optional[Detection] = None,
        track_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Path:
        validate_priority_gating(alert.priority, alert.zone, hypothesis)

        out_path = Path(output_dir)
        folder_name = alert.alert_id if alert.alert_id.startswith("alert_") else f"alert_{alert.alert_id}"
        package_dir = out_path / folder_name

        raw_dir = package_dir / "chips" / "raw"
        analysis_dir = package_dir / "chips" / "analysis"
        preview_dir = package_dir / "chips" / "preview"
        evidence_dir = package_dir / "evidence"

        raw_dir.mkdir(parents=True, exist_ok=True)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)
        evidence_dir.mkdir(parents=True, exist_ok=True)

        if "raw" in chips:
            raw_target = chips["raw"]
            if isinstance(raw_target, dict):
                for k, v in raw_target.items():
                    if isinstance(v, np.ndarray):
                        _write_raw_tiff(raw_dir / f"{k}.tif", v)
            elif isinstance(raw_target, np.ndarray):
                _write_raw_tiff(raw_dir / "raw.tif", raw_target)

            if "analysis" in chips:
                ana_target = chips["analysis"]
                if isinstance(ana_target, dict):
                    for k, v in ana_target.items():
                        if isinstance(v, np.ndarray):
                            _write_analysis_tensor(analysis_dir / k, v)
                elif isinstance(ana_target, np.ndarray):
                    _write_analysis_tensor(analysis_dir / "analysis", ana_target)
            elif isinstance(raw_target, np.ndarray):
                _write_analysis_tensor(analysis_dir / "analysis", raw_target)
            elif isinstance(raw_target, dict):
                for k, v in raw_target.items():
                    if isinstance(v, np.ndarray):
                        _write_analysis_tensor(analysis_dir / k, v)

            if "preview" in chips:
                prev_target = chips["preview"]
                if isinstance(prev_target, dict):
                    for k, v in prev_target.items():
                        if isinstance(v, np.ndarray):
                            _write_preview_png(preview_dir / f"{k}.png", v)
                elif isinstance(prev_target, np.ndarray):
                    _write_preview_png(preview_dir / "preview.png", prev_target)
            elif isinstance(raw_target, np.ndarray):
                _write_preview_png(preview_dir / "preview.png", raw_target)
            elif isinstance(raw_target, dict):
                for k, v in raw_target.items():
                    if isinstance(v, np.ndarray):
                        _write_preview_png(preview_dir / f"{k}.png", v)
        else:
            for k, v in chips.items():
                if isinstance(v, np.ndarray):
                    _write_raw_tiff(raw_dir / f"{k}.tif", v)
                    _write_analysis_tensor(analysis_dir / k, v)
                    _write_preview_png(preview_dir / f"{k}.png", v)

        manifest_data = {
            "alert_id": alert.alert_id,
            "hypothesis_id": alert.hypothesis_id,
            "schema_version": "2.0.0",
            "version": "2.0.0",
            "pipeline_version": alert.provenance.pipeline_version,
            "creation_time_utc": datetime.now(timezone.utc).isoformat(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "zone": alert.zone.value if hasattr(alert.zone, "value") else str(alert.zone),
            "priority": alert.priority.value if hasattr(alert.priority, "value") else str(alert.priority),
            "review_state": alert.review_state.value if hasattr(alert.review_state, "value") else str(alert.review_state),
            "operator_review_status": alert.review_state.value if hasattr(alert.review_state, "value") else str(alert.review_state),
            "review_status": alert.review_state.value if hasattr(alert.review_state, "value") else str(alert.review_state),
            "bundle_uri": alert.bundle_uri,
            "provenance_id": alert.provenance.provenance_id,
            "hypothesis": {
                "hypothesis_id": hypothesis.hypothesis_id,
                "statement": hypothesis.statement.value if hasattr(hypothesis.statement, "value") else str(hypothesis.statement),
                "score": hypothesis.score,
                "score_method": hypothesis.score_method.value if hasattr(hypothesis.score_method, "value") else str(hypothesis.score_method),
                "calibration_state": hypothesis.calibration_state.value if hasattr(hypothesis.calibration_state, "value") else str(hypothesis.calibration_state),
                "explanation": hypothesis.explanation,
                "detailed_rationale": hypothesis.detailed_rationale,
            },
            "chips": {
                "raw": sorted([p.name for p in raw_dir.glob("*")]),
                "analysis": sorted([p.name for p in analysis_dir.glob("*")]),
                "preview": sorted([p.name for p in preview_dir.glob("*")]),
            },
        }
        with open(package_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        provenance_data = _serialize_provenance(alert.provenance)
        with open(package_dir / "provenance.json", "w", encoding="utf-8") as f:
            json.dump(provenance_data, f, indent=2)

        hypothesis_data = {
            "hypothesis_id": hypothesis.hypothesis_id,
            "statement": hypothesis.statement.value if hasattr(hypothesis.statement, "value") else str(hypothesis.statement),
            "score": hypothesis.score,
            "score_method": hypothesis.score_method.value if hasattr(hypothesis.score_method, "value") else str(hypothesis.score_method),
            "calibration_state": hypothesis.calibration_state.value if hasattr(hypothesis.calibration_state, "value") else str(hypothesis.calibration_state),
            "explanation": hypothesis.explanation,
            "detailed_rationale": hypothesis.detailed_rationale,
            "primary_explanation": hypothesis.primary_explanation,
            "evidence_ids": list(hypothesis.evidence_ids),
            "supporting_evidence_count": len(hypothesis.supporting_evidence),
            "refuting_evidence_count": len(hypothesis.refuting_evidence),
            "provenance": _serialize_provenance(hypothesis.provenance),
        }
        with open(evidence_dir / "hypothesis.json", "w", encoding="utf-8") as f:
            json.dump(hypothesis_data, f, indent=2)

        supporting_data = [_serialize_evidence(e) for e in hypothesis.supporting_evidence]
        with open(evidence_dir / "supporting_evidence.json", "w", encoding="utf-8") as f:
            json.dump(supporting_data, f, indent=2)

        refuting_data = [_serialize_evidence(e) for e in hypothesis.refuting_evidence]
        with open(evidence_dir / "refuting_evidence.json", "w", encoding="utf-8") as f:
            json.dump(refuting_data, f, indent=2)

        if detection is not None:
            geom_dict = shapely.geometry.mapping(detection.geom_wgs84) if hasattr(detection, "geom_wgs84") else None
            bbox_list = list(detection.geom_wgs84.bounds) if hasattr(detection, "geom_wgs84") and hasattr(detection.geom_wgs84, "bounds") else None
            props = {
                "detection_id": detection.detection_id,
                "candidate_id": detection.candidate_id,
                "alert_id": alert.alert_id,
                "hypothesis_id": alert.hypothesis_id,
                "sensor": detection.sensor.value if hasattr(detection.sensor, "value") else str(detection.sensor),
                "target_class": detection.target_class.value if hasattr(detection.target_class, "value") else str(detection.target_class),
                "geom_px": list(detection.geom_px) if hasattr(detection, "geom_px") else None,
                "crs": detection.crs if hasattr(detection, "crs") else "EPSG:4326",
                "zone": alert.zone.value if hasattr(alert.zone, "value") else str(alert.zone),
                "track_id": track_id or kwargs.get("track_id") or kwargs.get("track_ref"),
                "confidence": detection.confidence.to_dict() if hasattr(detection.confidence, "to_dict") else None,
                "attributes": dict(detection.attributes) if hasattr(detection, "attributes") and detection.attributes else {},
                "acquisition_time_utc": detection.acquisition_time_utc.isoformat() if hasattr(detection, "acquisition_time_utc") and detection.acquisition_time_utc else None,
            }
        else:
            geom_dict = kwargs.get("geometry", {"type": "Point", "coordinates": [0.0, 0.0]})
            bbox_list = kwargs.get("bbox", [0.0, 0.0, 0.0, 0.0])
            props = {
                "alert_id": alert.alert_id,
                "hypothesis_id": alert.hypothesis_id,
                "zone": alert.zone.value if hasattr(alert.zone, "value") else str(alert.zone),
                "track_id": track_id or kwargs.get("track_id") or kwargs.get("track_ref"),
                "target_class": kwargs.get("target_class", "UNIDENTIFIED"),
                "sensor": kwargs.get("sensor", "S1_SAR"),
                "bbox": bbox_list,
                "attributes": kwargs.get("attributes", {}),
            }

        detection_feature = {
            "type": "Feature",
            "id": props.get("detection_id", f"det_{alert.alert_id}"),
            "geometry": geom_dict,
            "bbox": bbox_list,
            "properties": props,
        }

        geojson_data = {
            "type": "FeatureCollection",
            "features": [detection_feature],
        }
        with open(package_dir / "detection.geojson", "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=2)

        return package_dir
