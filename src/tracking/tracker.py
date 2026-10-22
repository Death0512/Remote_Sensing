from datetime import datetime, timezone
import math
from typing import Any
import uuid
import numpy as np
import pyproj

from src.domain.detection import Detection
from src.domain.enums import Capability, TrackLifecycle
from src.domain.provenance import Provenance
from src.domain.track import Track
from src.tracking.association.hungarian import GatedHungarianAssociator
from src.tracking.filters.kalman import KalmanFilter2D


class TrackState:
    def __init__(
        self,
        track_id: str,
        initial_detection: Detection,
        initial_xy: tuple[float, float],
        initial_time: datetime,
    ) -> None:
        self.track_id = track_id
        self.detections: list[Detection] = [initial_detection]
        self.last_update_time = initial_time
        self.hits = 1
        self.misses = 0
        self.lifecycle = TrackLifecycle.TENTATIVE
        self.kf = KalmanFilter2D(
            initial_state=np.array([initial_xy[0], initial_xy[1], 0.0, 0.0]),
            initial_covariance=np.diag([25.0, 25.0, 100.0, 100.0]),
            process_noise_std=1.0,
            measurement_noise_std=5.0,
        )
        self.predicted_pos = (float(initial_xy[0]), float(initial_xy[1]))

    def predict(self, dt: float) -> tuple[float, float]:
        if dt > 0.0:
            state, _ = self.kf.predict(dt)
            self.predicted_pos = (float(state[0, 0]), float(state[1, 0]))
        return self.predicted_pos

    def update_matched(self, detection: Detection, xy: tuple[float, float], timestamp: datetime) -> None:
        self.kf.update(np.array([xy[0], xy[1]]))
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
        gate_distance_m: float = 5000.0,
        max_dt_seconds: float = 7200.0,
        max_coast_misses: int = 2,
        center_lat: float = 16.0,
        center_lon: float = 108.0,
    ) -> None:
        self.gate_distance_m = gate_distance_m
        self.max_dt_seconds = max_dt_seconds
        self.max_coast_misses = max_coast_misses
        self.associator = GatedHungarianAssociator()
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

    def update(self, detections: list[Detection], current_time_utc: datetime) -> list[Track]:
        current_time = self._ensure_utc(current_time_utc)

        existing_track_ids = list(self.active_tracks.keys())
        active_candidates: list[str] = []

        for tid in existing_track_ids:
            track_state = self.active_tracks[tid]
            dt = (current_time - track_state.last_update_time).total_seconds()
            if dt > self.max_dt_seconds:
                track_state.lifecycle = TrackLifecycle.TERMINATED
            elif track_state.lifecycle != TrackLifecycle.TERMINATED:
                track_state.predict(dt)
                active_candidates.append(tid)

        det_objects: list[dict[str, Any]] = []
        for det in detections:
            xy = self._geom_to_xy(det.geom_wgs84)
            det_objects.append({"det": det, "xy": xy, "pos": xy})

        candidate_states = [self.active_tracks[tid] for tid in active_candidates]
        matched, unmatched_trks, unmatched_dets = self.associator.associate(
            candidate_states,
            det_objects,
            self.gate_distance_m,
        )

        for trk_idx, det_idx in matched:
            tid = active_candidates[trk_idx]
            det_info = det_objects[det_idx]
            self.active_tracks[tid].update_matched(
                detection=det_info["det"],
                xy=det_info["xy"],
                timestamp=current_time,
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

        latest_det = track_state.detections[-1]
        prov = Provenance(
            provenance_id=f"prov-trk-{uuid.uuid4().hex[:8]}",
            source_asset_id=latest_det.provenance.source_asset_id,
            acquisition_metadata_ref=latest_det.provenance.acquisition_metadata_ref,
            pipeline_version="2.0.0",
            git_commit="fedcba9",
            config_hash="cfg-tracker-v2",
            model_id=None,
            model_version=None,
            input_checksum=latest_det.provenance.output_checksum,
            output_checksum=f"chk-{uuid.uuid4().hex[:8]}",
            processing_run_id=latest_det.provenance.processing_run_id,
            processing_time_utc=current_time,
            parent_provenance_id=latest_det.provenance.provenance_id,
        )

        return Track(
            track_id=track_state.track_id,
            detection_ids=tuple(d.detection_id for d in track_state.detections),
            state_estimates=state_estimates,
            track_confidence=track_confidence,
            lifecycle=track_state.lifecycle,
            provenance=prov,
        )
