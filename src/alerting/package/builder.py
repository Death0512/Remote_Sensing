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
        "schema_version": getattr(prov, "schema_version", "provenance.v3"),
    }


def _serialize_confidence(conf: Any) -> Optional[dict[str, Any]]:
    if conf is None:
        return None
    if hasattr(conf, "to_dict"):
        return conf.to_dict()
    if isinstance(conf, dict):
        return conf
    if hasattr(conf, "raw_score"):
        return {
            "raw_score": conf.raw_score,
            "model_score": getattr(conf, "model_score", None),
            "calibrated_probability": getattr(conf, "calibrated_probability", None),
            "uncertainty": getattr(conf, "uncertainty", None),
            "quality_score": getattr(conf, "quality_score", 1.0),
            "calibration_state": conf.calibration_state.value if hasattr(getattr(conf, "calibration_state", None), "value") else str(getattr(conf, "calibration_state", None)),
            "schema_version": getattr(conf, "schema_version", "confidence_vector.v3"),
        }
    return {"raw_score": float(conf)}


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
    float_arr = arr.astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    if float_arr.ndim == 2:
        h, w = float_arr.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(
                str(path),
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=1,
                dtype="float32",
            ) as dst:
                dst.write(float_arr, 1)
    elif float_arr.ndim == 3:
        if float_arr.shape[0] in (1, 3, 4) and float_arr.shape[0] < float_arr.shape[1]:
            channels, h, w = float_arr.shape
            bands_data = float_arr
        elif float_arr.shape[2] in (1, 3, 4):
            h, w, channels = float_arr.shape
            bands_data = np.transpose(float_arr, (2, 0, 1))
        else:
            channels, h, w = float_arr.shape
            bands_data = float_arr
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(
                str(path),
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=channels,
                dtype="float32",
            ) as dst:
                dst.write(bands_data)
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")


def _write_preview_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    float_arr = arr.astype(np.float32)
    if float_arr.ndim == 3 and float_arr.shape[0] in (1, 3, 4) and float_arr.shape[0] < float_arr.shape[1]:
        float_arr = np.transpose(float_arr, (1, 2, 0))
    if float_arr.ndim == 3 and float_arr.shape[2] == 1:
        float_arr = float_arr[:, :, 0]

    if float_arr.size == 0:
        stretched = np.zeros((1, 1), dtype=np.uint8)
    else:
        p2 = float(np.nanpercentile(float_arr, 2))
        p98 = float(np.nanpercentile(float_arr, 98))
        if p98 > p2:
            stretched = np.clip((float_arr - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
        else:
            stretched = np.clip(float_arr, 0, 255).astype(np.uint8)
    Image.fromarray(stretched).save(path.with_suffix(".png"))


def _write_analysis_tensor(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    norm_arr = arr.astype(np.float32)
    min_val = float(np.nanmin(norm_arr)) if norm_arr.size > 0 else 0.0
    max_val = float(np.nanmax(norm_arr)) if norm_arr.size > 0 else 1.0
    if max_val > 1.0 or min_val < 0.0:
        if max_val > min_val:
            norm_arr = (norm_arr - min_val) / (max_val - min_val)
        else:
            norm_arr = np.zeros_like(norm_arr)
    np.save(path.with_suffix(".npy"), norm_arr)
    _write_raw_tiff(path.with_suffix(".tif"), norm_arr)


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
        schema_version: Optional[str] = None,
        **kwargs: Any,
    ) -> Path:
        validate_priority_gating(alert.priority, alert.zone, hypothesis)

        out_path = Path(output_dir)
        folder_name = alert.alert_id if alert.alert_id.startswith("alert_") else f"alert_{alert.alert_id}"
        package_dir = out_path if out_path.name == folder_name else out_path / folder_name

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

        if schema_version is not None:
            resolved_schema = str(schema_version)
        elif kwargs.get("schema_version") is not None:
            resolved_schema = str(kwargs["schema_version"])
        elif alert.provenance and alert.provenance.pipeline_version == "2.0.0":
            resolved_schema = "2.0.0"
        else:
            resolved_schema = "3.0.0"

        if alert.provenance and alert.provenance.pipeline_version:
            pipeline_version = alert.provenance.pipeline_version
        else:
            pipeline_version = "3.0.0"

        manifest_data = {
            "alert_id": alert.alert_id,
            "hypothesis_id": alert.hypothesis_id,
            "schema_version": resolved_schema,
            "version": resolved_schema,
            "package_schema": "alert_package.v3",
            "package_format": "alert_package.v3",
            "pipeline_version": pipeline_version,
            "creation_time_utc": datetime.now(timezone.utc).isoformat(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "zone": alert.zone.value if hasattr(alert.zone, "value") else str(alert.zone),
            "priority": alert.priority.value if hasattr(alert.priority, "value") else str(alert.priority),
            "review_state": alert.review_state.value if hasattr(alert.review_state, "value") else str(alert.review_state),
            "operator_review_status": alert.review_state.value if hasattr(alert.review_state, "value") else str(alert.review_state),
            "review_status": alert.review_state.value if hasattr(alert.review_state, "value") else str(alert.review_state),
            "bundle_uri": alert.bundle_uri,
            "provenance_id": alert.provenance.provenance_id if alert.provenance else None,
            "chips": {
                "raw": sorted([p.name for p in raw_dir.glob("*")]),
                "analysis": sorted([p.name for p in analysis_dir.glob("*")]),
                "preview": sorted([p.name for p in preview_dir.glob("*")]),
            },
            "hypothesis": {
                "hypothesis_id": hypothesis.hypothesis_id,
                "statement": hypothesis.statement.value if hasattr(hypothesis.statement, "value") else str(hypothesis.statement),
                "score": hypothesis.score,
                "score_method": hypothesis.score_method.value if hasattr(hypothesis.score_method, "value") else str(hypothesis.score_method),
                "calibration_state": hypothesis.calibration_state.value if hasattr(hypothesis.calibration_state, "value") else str(hypothesis.calibration_state),
                "explanation": hypothesis.explanation,
                "detailed_rationale": hypothesis.detailed_rationale,
            },
            "provenance": _serialize_provenance(alert.provenance) if alert.provenance else None,
            "provenance_summary": {
                "provenance_id": alert.provenance.provenance_id if alert.provenance else None,
                "pipeline_version": alert.provenance.pipeline_version if alert.provenance else None,
                "git_commit": alert.provenance.git_commit if alert.provenance else None,
                "config_hash": alert.provenance.config_hash if alert.provenance else None,
                "input_checksum": alert.provenance.input_checksum if alert.provenance else None,
                "output_checksum": alert.provenance.output_checksum if alert.provenance else None,
            } if alert.provenance else None,
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
            "primary_explanation": getattr(hypothesis, "primary_explanation", hypothesis.explanation),
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
            geom_wgs = detection.geom_wgs84 if getattr(detection, "geom_wgs84", None) is not None else getattr(detection, "geometry", None)
            geom_dict = shapely.geometry.mapping(geom_wgs) if geom_wgs is not None else None
            bbox_list = list(geom_wgs.bounds) if geom_wgs is not None and hasattr(geom_wgs, "bounds") else None
            conf_data = _serialize_confidence(detection.confidence)
            props = {
                "detection_id": detection.detection_id,
                "candidate_id": getattr(detection, "candidate_id", None),
                "alert_id": alert.alert_id,
                "hypothesis_id": alert.hypothesis_id,
                "sensor": detection.sensor.value if hasattr(detection.sensor, "value") else str(detection.sensor),
                "target_class": detection.target_class.value if hasattr(detection.target_class, "value") else str(detection.target_class),
                "geom_px": list(detection.geom_px) if hasattr(detection, "geom_px") and detection.geom_px is not None else None,
                "crs": detection.crs if hasattr(detection, "crs") else "EPSG:4326",
                "zone": alert.zone.value if hasattr(alert.zone, "value") else str(alert.zone),
                "track_id": track_id or kwargs.get("track_id") or kwargs.get("track_ref"),
                "confidence": conf_data,
                "attributes": dict(detection.attributes) if hasattr(detection, "attributes") and detection.attributes else {},
                "acquisition_time_utc": detection.acquisition_time_utc.isoformat() if hasattr(detection, "acquisition_time_utc") and detection.acquisition_time_utc else None,
            }
        else:
            geom_dict = kwargs.get("geometry", {"type": "Point", "coordinates": [0.0, 0.0]})
            bbox_list = kwargs.get("bbox", [0.0, 0.0, 0.0, 0.0])
            props = {
                "detection_id": kwargs.get("detection_id", f"det_{alert.alert_id}"),
                "alert_id": alert.alert_id,
                "hypothesis_id": alert.hypothesis_id,
                "zone": alert.zone.value if hasattr(alert.zone, "value") else str(alert.zone),
                "track_id": track_id or kwargs.get("track_id") or kwargs.get("track_ref"),
                "target_class": kwargs.get("target_class", "UNIDENTIFIED"),
                "sensor": kwargs.get("sensor", "S1_SAR"),
                "confidence": _serialize_confidence(kwargs.get("confidence")),
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
