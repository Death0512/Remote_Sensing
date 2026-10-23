from datetime import datetime, timezone
import math
from typing import Any, Optional
import uuid
import numpy as np
import pyproj
from shapely.geometry import Point

from src.domain.detection import Detection
from src.domain.enums import Capability, GeometryStatus, MotionClass, TrackLifecycle
from src.domain.provenance import Provenance
from src.domain.track import Track
from src.provenance.engine import compute_bytes_checksum, get_git_commit
from src.tracking.association.hungarian import MahalanobisAssociator
from src.tracking.filters.kalman import KalmanFilter2D
from src.tracking.motion_models import MotionModelConfig, get_motion_model, target_class_to_motion_class


class TrackState:
    def __init__(
        self,
        track_id: str,
        initial_detection: Detection,
        initial_xy: tuple[float, float],
        initial_time: datetime,
        motion_class: Optional[MotionClass] = None,
    ) -> None:
        self.track_id = track_id
        self.detections: list[Detection] = [initial_detection]
        self.last_update_time = initial_time
        self.hits = 1
        self.misses = 0
        self.lifecycle = TrackLifecycle.TENTATIVE

        if motion_class is not None:
            self.motion_class = motion_class
        else:
            target_cls = getattr(initial_detection, "target_class", None)
            self.motion_class = target_class_to_motion_class(target_cls)

        self.motion_model = get_motion_model(self.motion_class)
        initial_cov = self._get_initial_covariance()
        self.kf = KalmanFilter2D(
            initial_state=np.array([initial_xy[0], initial_xy[1], 0.0, 0.0]),
            initial_covariance=initial_cov,
            motion_model=self.motion_model,
        )
        self.predicted_pos = (float(initial_xy[0]), float(initial_xy[1]))

    def _get_initial_covariance(self) -> np.ndarray:
        meas_std = self.motion_model.default_measurement_noise_std
        pos_var = meas_std**2
        if self.motion_class == MotionClass.STATIONARY:
            return np.diag([pos_var, pos_var, 1e-4, 1e-4])
        if self.motion_class == MotionClass.VESSEL:
            return np.diag([pos_var, pos_var, 100.0, 100.0])
        if self.motion_class == MotionClass.AIRCRAFT:
            return np.diag([pos_var, pos_var, 2500.0, 2500.0])
        return np.diag([pos_var, pos_var, 100.0, 100.0])

    def predict(self, dt: float) -> tuple[float, float]:
        if dt > 0.0:
            state, _ = self.kf.predict(dt)
            self.predicted_pos = (float(state[0, 0]), float(state[1, 0]))
        return self.predicted_pos

    def update_matched(
        self,
        detection: Detection,
        xy: tuple[float, float],
        timestamp: datetime,
        measurement_cov: Optional[np.ndarray] = None,
    ) -> None:
        self.kf.update(np.array([xy[0], xy[1]]), measurement_cov=measurement_cov)
        self.detections.append(detection)
        self.last_update_time = timestamp
        self.hits += 1
        self.misses = 0
        self.lifecycle = TrackLifecycle.CONFIRMED
        self.predicted_pos = (float(self.kf.state[0, 0]), float(self.kf.state[1, 0]))

    def update_missed(self, max_coast_misses: int) -> None:
        self.misses += 1
        if self.misses > max_coast_misses or self.lifecycle == TrackLifecycle.TENTATIVE:
            self.lifecycle = TrackLifecycle.TERMINATED
        else:
            self.lifecycle = TrackLifecycle.COASTING


class MultiLookTracker:
    capability: Capability = Capability.IMPLEMENTED

    def __init__(
        self,
        gate_distance_m: Optional[float] = None,
        max_dt_seconds: Optional[float] = None,
        max_coast_misses: int = 2,
        center_lat: float = 16.0,
        center_lon: float = 108.0,
        associator: Optional[Any] = None,
    ) -> None:
        self.gate_distance_m = gate_distance_m
        self.max_dt_seconds = max_dt_seconds
        self.max_coast_misses = max_coast_misses
        self.associator = associator if associator is not None else MahalanobisAssociator()
        self.active_tracks: dict[str, TrackState] = {}
        self.aeqd_proj_string = (
            f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +datum=WGS84 +units=m +no_defs"
        )
        self.proj_aeqd = pyproj.CRS(self.aeqd_proj_string)
        self.proj_wgs84 = pyproj.CRS("EPSG:4326")
        self.forward_transform = pyproj.Transformer.from_crs(
            self.proj_wgs84, self.proj_aeqd, always_xy=True
        ).transform
        self.backward_transform = pyproj.Transformer.from_crs(
            self.proj_aeqd, self.proj_wgs84, always_xy=True
        ).transform

    def _ensure_utc(self, dt: datetime) -> datetime:
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    def _geom_to_xy(self, geom: Any) -> tuple[float, float]:
        lon = float(geom.centroid.x)
        lat = float(geom.centroid.y)
        x, y = self.forward_transform(lon, lat)
        return float(x), float(y)

    def _xy_to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        lon, lat = self.backward_transform(x, y)
        return float(lon), float(lat)

    def _get_effective_max_dt(self, motion_class: MotionClass) -> float:
        model = get_motion_model(motion_class)
        if self.max_dt_seconds is not None:
            return min(self.max_dt_seconds, model.max_dt_seconds)
        return model.max_dt_seconds

    def update(self, detections: list[Detection], current_time_utc: datetime) -> list[Track]:
        current_time = self._ensure_utc(current_time_utc)

        existing_track_ids = list(self.active_tracks.keys())
        active_candidates: list[str] = []

        for tid in existing_track_ids:
            track_state = self.active_tracks[tid]
            dt = (current_time - track_state.last_update_time).total_seconds()
            effective_max_dt = self._get_effective_max_dt(track_state.motion_class)
            if dt > effective_max_dt:
                track_state.lifecycle = TrackLifecycle.TERMINATED
            elif track_state.lifecycle != TrackLifecycle.TERMINATED:
                track_state.predict(dt)
                active_candidates.append(tid)

        det_objects: list[dict[str, Any]] = []
        for det in detections:
            xy = self._geom_to_xy(det.geom_wgs84)
            time_unc = 0.0
            if hasattr(det, "time_uncertainty_seconds") and det.time_uncertainty_seconds is not None:
                time_unc = float(det.time_uncertainty_seconds)
            elif hasattr(det, "attributes") and det.attributes:
                time_unc = float(dict(det.attributes).get("time_uncertainty_seconds", 0.0))

            meas_cov = None
            if hasattr(det, "measurement_covariance") and det.measurement_covariance is not None:
                meas_cov = np.asarray(det.measurement_covariance, dtype=np.float64)
            elif hasattr(det, "attributes") and det.attributes:
                meas_std = dict(det.attributes).get("measurement_noise_std")
                if meas_std is not None:
                    meas_cov = np.eye(2, dtype=np.float64) * (float(meas_std) ** 2)

            det_objects.append(
                {
                    "det": det,
                    "xy": xy,
                    "pos": xy,
                    "time_uncertainty_seconds": time_unc,
                    "measurement_covariance": meas_cov,
                }
            )

        candidate_states = [self.active_tracks[tid] for tid in active_candidates]
        matched, unmatched_trks, unmatched_dets = self.associator.associate(
            candidate_states,
            det_objects,
            gate_distance_m=self.gate_distance_m,
        )

        for trk_idx, det_idx in matched:
            tid = active_candidates[trk_idx]
            det_info = det_objects[det_idx]
            self.active_tracks[tid].update_matched(
                detection=det_info["det"],
                xy=det_info["xy"],
                timestamp=current_time,
                measurement_cov=det_info["measurement_covariance"],
            )

        for trk_idx in unmatched_trks:
            tid = active_candidates[trk_idx]
            self.active_tracks[tid].update_missed(self.max_coast_misses)

        for det_idx in unmatched_dets:
            det_info = det_objects[det_idx]
            new_id = f"TRK-{uuid.uuid4().hex[:8]}"
            new_state = TrackState(
                track_id=new_id,
                initial_detection=det_info["det"],
                initial_xy=det_info["xy"],
                initial_time=current_time,
            )
            self.active_tracks[new_id] = new_state

        output_tracks: list[Track] = []
        for tid, track_state in list(self.active_tracks.items()):
            track_obj = self._to_domain_track(track_state, current_time)
            output_tracks.append(track_obj)
            if track_state.lifecycle == TrackLifecycle.TERMINATED:
                del self.active_tracks[tid]

        return output_tracks

    def _to_domain_track(self, track_state: TrackState, current_time: datetime) -> Track:
        state_vec = track_state.kf.state
        x = float(state_vec[0, 0])
        y = float(state_vec[1, 0])
        vx = float(state_vec[2, 0])
        vy = float(state_vec[3, 0])

        speed_mps = math.hypot(vx, vy)
        velocity_knots = speed_mps * 1.943844
        heading_deg = (math.degrees(math.atan2(vx, vy)) + 360.0) % 360.0

        lon, lat = self._xy_to_lonlat(x, y)

        cov_diag = np.diag(track_state.kf.covariance)
        pos_var = float(cov_diag[0] + cov_diag[1])
        track_confidence = max(0.1, min(0.99, 1.0 / (1.0 + pos_var / 1000.0)))
        quality = float(max(0.1, min(1.0, track_confidence * (1.0 - 0.1 * track_state.misses))))

        state_tuple = (x, y, vx, vy)
        cov_tuple = tuple(
            tuple(float(track_state.kf.covariance[r, c]) for c in range(4))
            for r in range(4)
        )
        det_ids = tuple(
            d.detection_id for d in track_state.detections if hasattr(d, "detection_id")
        )
        sensor_sources = tuple(
            sorted(
                set(
                    d.sensor.value if hasattr(d.sensor, "value") else str(d.sensor)
                    for d in track_state.detections
                    if hasattr(d, "sensor") and d.sensor is not None
                )
            )
        )

        state_estimates = (
            (
                ("x_m", x),
                ("y_m", y),
                ("longitude", lon),
                ("latitude", lat),
                ("vx_mps", vx),
                ("vy_mps", vy),
                ("velocity_knots", velocity_knots),
                ("heading_deg", heading_deg),
                ("hits", float(track_state.hits)),
                ("misses", float(track_state.misses)),
            ),
        )

        prov = self._create_provenance(track_state, current_time)

        return Track(
            track_id=track_state.track_id,
            detection_ids=det_ids,
            state_estimates=state_estimates,
            track_confidence=track_confidence,
            lifecycle=track_state.lifecycle,
            provenance=prov,
            schema_version="track.v3",
            geometry_status=GeometryStatus.GEOCODED,
            geometry=Point(lon, lat),
            motion_class=track_state.motion_class,
            state=state_tuple,
            state_covariance=cov_tuple,
            last_update_utc=track_state.last_update_time,
            prediction_time_utc=current_time,
            associated_detection_ids=det_ids,
            sensor_sources=sensor_sources,
            quality=quality,
        )

    def _create_provenance(self, track_state: TrackState, current_time: datetime) -> Provenance:
        latest_det = track_state.detections[-1]
        det_prov = getattr(latest_det, "provenance", None)
        source_asset = det_prov.source_asset_id if det_prov else "asset-track"
        meta_ref = det_prov.acquisition_metadata_ref if det_prov else "s3://meta/track.xml"
        proc_run_id = det_prov.processing_run_id if det_prov else f"run-trk-{uuid.uuid4().hex[:8]}"
        parent_id = det_prov.provenance_id if det_prov else None
        in_chk = det_prov.output_checksum if det_prov else compute_bytes_checksum(b"trk-input")
        out_bytes = (
            f"{track_state.track_id}:{track_state.hits}:{track_state.misses}:{track_state.kf.state.ravel().tolist()}"
        ).encode("utf-8")
        out_chk = compute_bytes_checksum(out_bytes)

        return Provenance(
            provenance_id=f"prov-trk-{uuid.uuid4().hex[:12]}",
            source_asset_id=source_asset,
            acquisition_metadata_ref=meta_ref,
            pipeline_version="3.0.0",
            git_commit=get_git_commit(),
            config_hash=compute_bytes_checksum(
                f"cfg-tracker-v3-{track_state.motion_class.value}".encode("utf-8")
            ),
            model_id=None,
            model_version=None,
            input_checksum=in_chk,
            output_checksum=out_chk,
            processing_run_id=proc_run_id,
            processing_time_utc=current_time,
            parent_provenance_id=parent_id,
        )
