from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
from typing import Any, Optional, Sequence, Union
import uuid

import pyproj
from shapely.geometry import Point

from src.domain.association import Association
from src.domain.detection import Detection
from src.domain.enums import AisCorrelationState, Capability, EvidenceKind, GeometryStatus
from src.domain.evidence import Evidence
from src.domain.provenance import Provenance
from src.provenance.engine import compute_bytes_checksum


@dataclass(frozen=True)
class AisRecord:
    mmsi: str
    lat: float
    lon: float
    timestamp: datetime
    vessel_name: Optional[str] = None
    vessel_type: Optional[str] = None
    speed_knots: Optional[float] = None
    heading_deg: Optional[float] = None
    pos_uncertainty_m: float = 20.0


@dataclass(frozen=True)
class AisCoverageZone:
    zone_id: str
    geometry: Any
    active: bool = True
    start_time_utc: Optional[datetime] = None
    end_time_utc: Optional[datetime] = None
    receiver_type: str = "TERRESTRIAL"


class AisCoverageModel:
    def __init__(
        self,
        zones: Sequence[AisCoverageZone] = (),
        default_covered: bool = False,
    ) -> None:
        self.zones = tuple(zones)
        self.default_covered = default_covered

    def is_covered(
        self,
        lon: float,
        lat: float,
        timestamp_utc: Optional[datetime] = None,
    ) -> bool:
        if not self.zones:
            return self.default_covered
        pt = Point(lon, lat)
        for zone in self.zones:
            if not zone.active:
                continue
            if timestamp_utc is not None:
                if zone.start_time_utc is not None and timestamp_utc < zone.start_time_utc:
                    continue
                if zone.end_time_utc is not None and timestamp_utc > zone.end_time_utc:
                    continue
            if zone.geometry.contains(pt) or zone.geometry.intersects(pt):
                return True
        return False


@dataclass(frozen=True)
class CandidateMatch:
    record: AisRecord
    predicted_lon: float
    predicted_lat: float
    residual_m: float
    gate_statistic: float
    propagated_uncertainty_m: float
    dt_seconds: float


class AisCorrelator:
    capability: Capability = Capability.IMPLEMENTED

    def __init__(
        self,
        coverage_model: Optional[AisCoverageModel] = None,
        max_stale_seconds: float = 3600.0,
        speed_tolerance_knots: float = 5.0,
        heading_tolerance_deg: float = 45.0,
        default_max_speed_mps: float = 15.43,
        pipeline_version: str = "3.0.0",
    ) -> None:
        self.coverage_model = coverage_model
        self.max_stale_seconds = max_stale_seconds
        self.speed_tolerance_knots = speed_tolerance_knots
        self.heading_tolerance_deg = heading_tolerance_deg
        self.default_max_speed_mps = default_max_speed_mps
        self.pipeline_version = pipeline_version
        self.geod = pyproj.Geod(ellps="WGS84")

    def _ensure_utc(self, dt: datetime) -> datetime:
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _extract_coords(self, detection: Detection) -> tuple[float, float]:
        centroid = detection.geom_wgs84.centroid
        return float(centroid.x), float(centroid.y)

    def _extract_detection_uncertainty(self, detection: Detection) -> float:
        if hasattr(detection, "uncertainty") and detection.uncertainty is not None:
            return float(detection.uncertainty)
        attrs = dict(detection.attributes) if hasattr(detection, "attributes") else {}
        if "uncertainty_m" in attrs:
            return float(attrs["uncertainty_m"])
        return 30.0

    def _create_provenance(
        self,
        detection: Detection,
        association_id: str,
        target_id: str,
    ) -> Provenance:
        input_data = f"{detection.detection_id}:{target_id}".encode("utf-8")
        output_data = association_id.encode("utf-8")
        parent_id = detection.provenance.provenance_id if detection.provenance else None
        source_asset_id = (
            detection.provenance.source_asset_id
            if detection.provenance
            else "AIS_CORRELATOR"
        )
        metadata_ref = (
            detection.provenance.acquisition_metadata_ref
            if detection.provenance
            else "s3://ais/telemetry"
        )
        git_commit = (
            detection.provenance.git_commit
            if detection.provenance
            else "fedcba9"
        )
        proc_run_id = (
            detection.provenance.processing_run_id
            if detection.provenance
            else f"run-ais-{uuid.uuid4().hex[:8]}"
        )

        return Provenance(
            provenance_id=str(uuid.uuid4()),
            source_asset_id=source_asset_id,
            acquisition_metadata_ref=metadata_ref,
            pipeline_version=self.pipeline_version,
            git_commit=git_commit,
            config_hash=compute_bytes_checksum(
                f"ais-correlator-stale={self.max_stale_seconds}".encode("utf-8")
            ),
            model_id=None,
            model_version=None,
            input_checksum=compute_bytes_checksum(input_data),
            output_checksum=compute_bytes_checksum(output_data),
            processing_run_id=proc_run_id,
            processing_time_utc=datetime.now(timezone.utc),
            parent_provenance_id=parent_id,
        )

    def propagate_uncertainty(
        self,
        sigma_meas: float,
        sigma_ais: float,
        dt_seconds: float,
        v_max_mps: Optional[float] = None,
    ) -> float:
        v_max = v_max_mps if v_max_mps is not None else self.default_max_speed_mps
        kinematic_dispersion = (v_max * dt_seconds) ** 2
        total_var = (sigma_meas**2) + (sigma_ais**2) + kinematic_dispersion
        return math.sqrt(total_var)

    def predict_position(
        self,
        record: AisRecord,
        target_time: datetime,
    ) -> tuple[float, float]:
        rec_time = self._ensure_utc(record.timestamp)
        t_time = self._ensure_utc(target_time)
        dt_sec = (t_time - rec_time).total_seconds()

        if abs(dt_sec) < 1e-3 or record.speed_knots is None or record.heading_deg is None:
            return record.lon, record.lat

        speed_mps = record.speed_knots * 0.514444
        distance_m = speed_mps * abs(dt_sec)

        if dt_sec >= 0:
            azimuth = record.heading_deg % 360.0
        else:
            azimuth = (record.heading_deg + 180.0) % 360.0

        pred_lon, pred_lat, _ = self.geod.fwd(record.lon, record.lat, azimuth, distance_m)
        return float(pred_lon), float(pred_lat)

    def associate(
        self,
        detection: Detection,
        ais_records: Sequence[AisRecord],
        feed_active: bool = True,
        coverage_model: Optional[AisCoverageModel] = None,
        last_feed_time_utc: Optional[datetime] = None,
        spatial_gate_m: float = 3000.0,
        temporal_gate_s: float = 1800.0,
        max_gate_sigma: float = 3.0,
        max_uncertainty_threshold_m: float = 25000.0,
    ) -> Association:
        det_time = self._ensure_utc(detection.acquisition_time_utc)
        det_lon, det_lat = self._extract_coords(detection)
        sigma_meas = self._extract_detection_uncertainty(detection)
        assoc_id = f"assoc-{uuid.uuid4().hex[:12]}"

        cov_model = coverage_model if coverage_model is not None else self.coverage_model

        if not feed_active:
            prov = self._create_provenance(detection, assoc_id, "NONE")
            return Association(
                association_id=assoc_id,
                source_id=detection.detection_id,
                target_id="NONE",
                association_type=AisCorrelationState.AIS_UNAVAILABLE.value,
                residual=0.0,
                gate_statistic=0.0,
                confidence=0.0,
                provenance=prov,
                surviving_uncertainty=10000.0,
                quality_flags=("FEED_INACTIVE", "AIS_UNAVAILABLE"),
                geometry=Point(det_lon, det_lat),
                geometry_status=GeometryStatus.GEOCODED,
                timestamp_utc=det_time,
            )

        if cov_model is None:
            prov = self._create_provenance(detection, assoc_id, "NONE")
            return Association(
                association_id=assoc_id,
                source_id=detection.detection_id,
                target_id="NONE",
                association_type=AisCorrelationState.AIS_UNAVAILABLE.value,
                residual=0.0,
                gate_statistic=0.0,
                confidence=0.0,
                provenance=prov,
                surviving_uncertainty=10000.0,
                quality_flags=("NO_COVERAGE_MODEL", "AIS_UNAVAILABLE"),
                geometry=Point(det_lon, det_lat),
                geometry_status=GeometryStatus.GEOCODED,
                timestamp_utc=det_time,
            )

        if not cov_model.is_covered(det_lon, det_lat, det_time):
            prov = self._create_provenance(detection, assoc_id, "NONE")
            return Association(
                association_id=assoc_id,
                source_id=detection.detection_id,
                target_id="NONE",
                association_type=AisCorrelationState.AIS_UNAVAILABLE.value,
                residual=0.0,
                gate_statistic=0.0,
                confidence=0.0,
                provenance=prov,
                surviving_uncertainty=10000.0,
                quality_flags=("OUTSIDE_COVERAGE", "AIS_UNAVAILABLE"),
                geometry=Point(det_lon, det_lat),
                geometry_status=GeometryStatus.GEOCODED,
                timestamp_utc=det_time,
            )

        if last_feed_time_utc is not None:
            last_feed_utc = self._ensure_utc(last_feed_time_utc)
            stale_delta = abs((det_time - last_feed_utc).total_seconds())
            if stale_delta > self.max_stale_seconds:
                prov = self._create_provenance(detection, assoc_id, "NONE")
                return Association(
                    association_id=assoc_id,
                    source_id=detection.detection_id,
                    target_id="NONE",
                    association_type=AisCorrelationState.AIS_STALE.value,
                    residual=0.0,
                    gate_statistic=0.0,
                    confidence=0.0,
                    provenance=prov,
                    surviving_uncertainty=5000.0,
                    quality_flags=("FEED_STALE", f"stale_seconds={stale_delta:.1f}"),
                    geometry=Point(det_lon, det_lat),
                    geometry_status=GeometryStatus.GEOCODED,
                    timestamp_utc=det_time,
                )

        dedup_records: dict[str, AisRecord] = {}
        for rec in ais_records:
            rec_t = self._ensure_utc(rec.timestamp)
            dt = abs((det_time - rec_t).total_seconds())
            if dt > temporal_gate_s:
                continue
            if rec.mmsi not in dedup_records:
                dedup_records[rec.mmsi] = rec
            else:
                existing_dt = abs((det_time - self._ensure_utc(dedup_records[rec.mmsi].timestamp)).total_seconds())
                if dt < existing_dt:
                    dedup_records[rec.mmsi] = rec

        candidates: list[CandidateMatch] = []
        for rec in dedup_records.values():
            rec_t = self._ensure_utc(rec.timestamp)
            dt = abs((det_time - rec_t).total_seconds())

            v_eff = self.default_max_speed_mps
            if rec.speed_knots is not None:
                v_eff = max(v_eff, rec.speed_knots * 0.514444)

            sigma_pos = self.propagate_uncertainty(
                sigma_meas=sigma_meas,
                sigma_ais=rec.pos_uncertainty_m,
                dt_seconds=dt,
                v_max_mps=v_eff,
            )

            pred_lon, pred_lat = self.predict_position(rec, det_time)
            _, _, dist = self.geod.inv(det_lon, det_lat, pred_lon, pred_lat)
            residual_m = float(dist)

            gate_stat = residual_m / max(sigma_pos, 1.0)
            effective_spatial_gate = max(spatial_gate_m, max_gate_sigma * sigma_pos)

            if residual_m <= effective_spatial_gate:
                candidates.append(
                    CandidateMatch(
                        record=rec,
                        predicted_lon=pred_lon,
                        predicted_lat=pred_lat,
                        residual_m=residual_m,
                        gate_statistic=gate_stat,
                        propagated_uncertainty_m=sigma_pos,
                        dt_seconds=dt,
                    )
                )

        if not candidates:
            prov = self._create_provenance(detection, assoc_id, "NONE")
            return Association(
                association_id=assoc_id,
                source_id=detection.detection_id,
                target_id="NONE",
                association_type=AisCorrelationState.NO_MATCH.value,
                residual=0.0,
                gate_statistic=0.0,
                confidence=0.0,
                provenance=prov,
                surviving_uncertainty=round(sigma_meas, 2),
                quality_flags=(
                    "COVERAGE_VERIFIED",
                    "AIS_NO_MATCH_WITHIN_VERIFIED_COVERAGE",
                ),
                geometry=Point(det_lon, det_lat),
                geometry_status=GeometryStatus.GEOCODED,
                timestamp_utc=det_time,
            )

        candidates.sort(key=lambda c: (c.gate_statistic, c.residual_m))

        if len(candidates) > 1:
            best_c = candidates[0]
            second_c = candidates[1]
            close_residual = second_c.residual_m <= (best_c.residual_m * 1.5 + 50.0)
            close_gate = abs(second_c.gate_statistic - best_c.gate_statistic) < 0.6
            if close_residual or close_gate:
                prov = self._create_provenance(detection, assoc_id, best_c.record.mmsi)
                mmsis = tuple(c.record.mmsi for c in candidates[:3])
                return Association(
                    association_id=assoc_id,
                    source_id=detection.detection_id,
                    target_id=best_c.record.mmsi,
                    association_type=AisCorrelationState.CORRELATION_UNCERTAIN.value,
                    residual=round(best_c.residual_m, 2),
                    gate_statistic=round(best_c.gate_statistic, 3),
                    confidence=0.45,
                    provenance=prov,
                    surviving_uncertainty=round(best_c.propagated_uncertainty_m, 2),
                    quality_flags=(
                        "AMBIGUOUS_MULTI_CANDIDATE",
                        "COVERAGE_VERIFIED",
                        *(f"candidate_mmsi={m}" for m in mmsis),
                    ),
                    geometry=Point(best_c.predicted_lon, best_c.predicted_lat),
                    geometry_status=GeometryStatus.GEOCODED,
                    timestamp_utc=det_time,
                )

        best = candidates[0]
        if best.propagated_uncertainty_m > max_uncertainty_threshold_m:
            prov = self._create_provenance(detection, assoc_id, best.record.mmsi)
            return Association(
                association_id=assoc_id,
                source_id=detection.detection_id,
                target_id=best.record.mmsi,
                association_type=AisCorrelationState.CORRELATION_UNCERTAIN.value,
                residual=round(best.residual_m, 2),
                gate_statistic=round(best.gate_statistic, 3),
                confidence=0.4,
                provenance=prov,
                surviving_uncertainty=round(best.propagated_uncertainty_m, 2),
                quality_flags=(
                    "HIGH_PROPAGATED_UNCERTAINTY",
                    "COVERAGE_VERIFIED",
                ),
                geometry=Point(best.predicted_lon, best.predicted_lat),
                geometry_status=GeometryStatus.GEOCODED,
                timestamp_utc=det_time,
            )

        det_attrs = dict(detection.attributes) if hasattr(detection, "attributes") else {}
        det_speed = det_attrs.get("speed_knots")
        det_heading = det_attrs.get("heading_deg")

        is_kinematic_mismatch = False
        if det_speed is not None and best.record.speed_knots is not None:
            if abs(float(det_speed) - float(best.record.speed_knots)) > self.speed_tolerance_knots:
                is_kinematic_mismatch = True

        if det_heading is not None and best.record.heading_deg is not None:
            heading_diff = abs(float(det_heading) - float(best.record.heading_deg)) % 360.0
            if heading_diff > 180.0:
                heading_diff = 360.0 - heading_diff
            if heading_diff > self.heading_tolerance_deg:
                is_kinematic_mismatch = True

        prov = self._create_provenance(detection, assoc_id, best.record.mmsi)

        if is_kinematic_mismatch:
            state = AisCorrelationState.PARTIAL_MATCH
            conf = 0.65
            flags = ("KINEMATIC_MISMATCH", "COVERAGE_VERIFIED")
        else:
            state = AisCorrelationState.MATCHED
            conf = round(max(0.75, 1.0 - (best.gate_statistic / max_gate_sigma) * 0.25), 3)
            flags = ("KINEMATIC_VERIFIED", "COVERAGE_VERIFIED")

        return Association(
            association_id=assoc_id,
            source_id=detection.detection_id,
            target_id=best.record.mmsi,
            association_type=state.value,
            residual=round(best.residual_m, 2),
            gate_statistic=round(best.gate_statistic, 3),
            confidence=conf,
            provenance=prov,
            surviving_uncertainty=round(best.propagated_uncertainty_m, 2),
            quality_flags=flags,
            geometry=Point(best.predicted_lon, best.predicted_lat),
            geometry_status=GeometryStatus.GEOCODED,
            timestamp_utc=det_time,
        )

    def correlate(
        self,
        detection: Detection,
        ais_records: Sequence[AisRecord],
        feed_active: bool = True,
        coverage_model: Optional[AisCoverageModel] = None,
        last_feed_time_utc: Optional[datetime] = None,
        spatial_gate_m: float = 3000.0,
        temporal_gate_s: float = 1800.0,
        max_gate_sigma: float = 3.0,
        max_uncertainty_threshold_m: float = 25000.0,
    ) -> Association:
        return self.associate(
            detection=detection,
            ais_records=ais_records,
            feed_active=feed_active,
            coverage_model=coverage_model,
            last_feed_time_utc=last_feed_time_utc,
            spatial_gate_m=spatial_gate_m,
            temporal_gate_s=temporal_gate_s,
            max_gate_sigma=max_gate_sigma,
            max_uncertainty_threshold_m=max_uncertainty_threshold_m,
        )

    def associate_batch(
        self,
        detections: Sequence[Detection],
        ais_records: Sequence[AisRecord],
        feed_active: bool = True,
        coverage_model: Optional[AisCoverageModel] = None,
        last_feed_time_utc: Optional[datetime] = None,
        spatial_gate_m: float = 3000.0,
        temporal_gate_s: float = 1800.0,
        max_gate_sigma: float = 3.0,
        max_uncertainty_threshold_m: float = 25000.0,
    ) -> tuple[Association, ...]:
        assigned_mmsis: set[str] = set()
        associations: list[Association] = []

        all_pair_candidates: list[tuple[float, int, AisRecord]] = []
        for det_idx, det in enumerate(detections):
            det_lon, det_lat = self._extract_coords(det)
            det_time = self._ensure_utc(det.acquisition_time_utc)
            sigma_meas = self._extract_detection_uncertainty(det)

            for rec in ais_records:
                rec_t = self._ensure_utc(rec.timestamp)
                dt = abs((det_time - rec_t).total_seconds())
                if dt > temporal_gate_s:
                    continue

                v_eff = self.default_max_speed_mps
                if rec.speed_knots is not None:
                    v_eff = max(v_eff, rec.speed_knots * 0.514444)

                sigma_pos = self.propagate_uncertainty(sigma_meas, rec.pos_uncertainty_m, dt, v_eff)
                pred_lon, pred_lat = self.predict_position(rec, det_time)
                _, _, dist = self.geod.inv(det_lon, det_lat, pred_lon, pred_lat)
                residual_m = float(dist)
                gate_stat = residual_m / max(sigma_pos, 1.0)
                eff_spatial = max(spatial_gate_m, max_gate_sigma * sigma_pos)

                if residual_m <= eff_spatial and gate_stat <= max_gate_sigma:
                    all_pair_candidates.append((gate_stat, det_idx, rec))

        all_pair_candidates.sort(key=lambda p: p[0])

        det_assigned_mmsi: dict[int, str] = {}
        for gate_stat, det_idx, rec in all_pair_candidates:
            if det_idx in det_assigned_mmsi:
                continue
            if rec.mmsi in assigned_mmsis:
                continue
            assigned_mmsis.add(rec.mmsi)
            det_assigned_mmsi[det_idx] = rec.mmsi

        for det_idx, det in enumerate(detections):
            if det_idx in det_assigned_mmsi:
                target_mmsi = det_assigned_mmsi[det_idx]
                target_records = [r for r in ais_records if r.mmsi == target_mmsi]
                assoc = self.associate(
                    detection=det,
                    ais_records=target_records,
                    feed_active=feed_active,
                    coverage_model=coverage_model,
                    last_feed_time_utc=last_feed_time_utc,
                    spatial_gate_m=spatial_gate_m,
                    temporal_gate_s=temporal_gate_s,
                    max_gate_sigma=max_gate_sigma,
                    max_uncertainty_threshold_m=max_uncertainty_threshold_m,
                )
            else:
                remaining_records = [r for r in ais_records if r.mmsi not in assigned_mmsis]
                assoc = self.associate(
                    detection=det,
                    ais_records=remaining_records,
                    feed_active=feed_active,
                    coverage_model=coverage_model,
                    last_feed_time_utc=last_feed_time_utc,
                    spatial_gate_m=spatial_gate_m,
                    temporal_gate_s=temporal_gate_s,
                    max_gate_sigma=max_gate_sigma,
                    max_uncertainty_threshold_m=max_uncertainty_threshold_m,
                )
            associations.append(assoc)

        return tuple(associations)

    def to_evidence(self, association: Association) -> Evidence:
        state_val = association.association_type
        unc_norm = (
            min(association.gate_statistic / 3.0, 1.0)
            if association.gate_statistic > 0.0
            else 0.05
        )
        if state_val == AisCorrelationState.AIS_UNAVAILABLE.value:
            unc_norm = 1.0
            val = 0.0
            qual = 0.0
        elif state_val == AisCorrelationState.AIS_STALE.value:
            unc_norm = 0.8
            val = 0.0
            qual = 0.2
        elif state_val == AisCorrelationState.NO_MATCH.value:
            val = 0.0
            qual = 0.95
        elif state_val == AisCorrelationState.PARTIAL_MATCH.value:
            val = association.confidence
            qual = 0.85
        elif state_val == AisCorrelationState.CORRELATION_UNCERTAIN.value:
            val = 0.5
            qual = 0.5
            unc_norm = 0.7
        else:
            val = association.confidence
            qual = 0.95

        ref_list = [
            f"state={state_val}",
            association.source_id,
            f"association_id={association.association_id}",
            f"target_id={association.target_id}",
            f"residual_m={association.residual:.2f}",
            f"gate_statistic={association.gate_statistic:.3f}",
            f"surviving_uncertainty_m={association.surviving_uncertainty:.2f}"
            if association.surviving_uncertainty is not None
            else "surviving_uncertainty_m=0.00",
        ]
        if "COVERAGE_VERIFIED" in association.quality_flags:
            ref_list.append("coverage=VERIFIED")
        if "FEED_INACTIVE" in association.quality_flags:
            ref_list.append("feed_active=False")

        for flag in association.quality_flags:
            ref_list.append(f"flag={flag}")

        return Evidence(
            evidence_id=f"ev-{association.association_id}",
            kind=EvidenceKind.AIS_CORRELATION,
            value=val,
            quality=qual,
            source="ais_correlator",
            timestamp_utc=association.timestamp_utc or datetime.now(timezone.utc),
            uncertainty=round(unc_norm, 3),
            supports=state_val,
            references=tuple(ref_list),
            provenance=association.provenance,
        )

    def correlate_to_evidence(
        self,
        detection: Detection,
        ais_records: Sequence[AisRecord],
        feed_active: bool = True,
        coverage_model: Optional[AisCoverageModel] = None,
        last_feed_time_utc: Optional[datetime] = None,
        spatial_gate_m: float = 3000.0,
        temporal_gate_s: float = 1800.0,
        max_gate_sigma: float = 3.0,
        max_uncertainty_threshold_m: float = 25000.0,
    ) -> Evidence:
        assoc = self.associate(
            detection=detection,
            ais_records=ais_records,
            feed_active=feed_active,
            coverage_model=coverage_model,
            last_feed_time_utc=last_feed_time_utc,
            spatial_gate_m=spatial_gate_m,
            temporal_gate_s=temporal_gate_s,
            max_gate_sigma=max_gate_sigma,
            max_uncertainty_threshold_m=max_uncertainty_threshold_m,
        )
        return self.to_evidence(assoc)
