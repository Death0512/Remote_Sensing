from typing import Any, Optional, Sequence
import numpy as np
from scipy.optimize import linear_sum_assignment


class GatedHungarianAssociator:
    def __init__(self, cost_gate_multiplier: float = 1e6) -> None:
        self.cost_gate_multiplier = cost_gate_multiplier

    def _extract_pos(self, obj: Any) -> tuple[float, float]:
        if isinstance(obj, dict):
            if "pos" in obj:
                return obj["pos"]
            if "predicted_pos" in obj:
                return obj["predicted_pos"]
        if hasattr(obj, "predicted_pos"):
            return getattr(obj, "predicted_pos")
        if hasattr(obj, "pos"):
            return getattr(obj, "pos")
        return obj

    def associate(
        self,
        tracks: Sequence[Any],
        detections: Sequence[Any],
        gate_distance_m: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        num_tracks = len(tracks)
        num_detections = len(detections)

        if num_tracks == 0:
            return [], [], list(range(num_detections))
        if num_detections == 0:
            return [], list(range(num_tracks)), []

        cost_matrix = np.full((num_tracks, num_detections), np.inf, dtype=np.float64)

        for i, track in enumerate(tracks):
            pos_t = self._extract_pos(track)
            for j, det in enumerate(detections):
                pos_d = self._extract_pos(det)
                dist = float(np.hypot(pos_t[0] - pos_d[0], pos_t[1] - pos_d[1]))
                if dist <= gate_distance_m:
                    cost_matrix[i, j] = dist

        large_val = gate_distance_m * self.cost_gate_multiplier
        solvable_cost = np.where(np.isinf(cost_matrix), large_val, cost_matrix)

        row_ind, col_ind = linear_sum_assignment(solvable_cost)

        matched_pairs: list[tuple[int, int]] = []
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= gate_distance_m:
                matched_pairs.append((int(r), int(c)))
                matched_tracks.add(int(r))
                matched_detections.add(int(c))

        unmatched_tracks = [i for i in range(num_tracks) if i not in matched_tracks]
        unmatched_detections = [j for j in range(num_detections) if j not in matched_detections]

        return matched_pairs, unmatched_tracks, unmatched_detections


class MahalanobisAssociator:
    def __init__(
        self,
        default_chi2_threshold: float = 9.21034,
        cost_gate_multiplier: float = 1e6,
    ) -> None:
        self.default_chi2_threshold = default_chi2_threshold
        self.cost_gate_multiplier = cost_gate_multiplier

    def _extract_track_state(self, track: Any) -> tuple[tuple[float, float], np.ndarray, tuple[float, float], float]:
        pos = (0.0, 0.0)
        cov = np.diag([25.0, 25.0, 100.0, 100.0])
        vel = (0.0, 0.0)
        gate = self.default_chi2_threshold

        if hasattr(track, "predicted_pos"):
            p = getattr(track, "predicted_pos")
            pos = (float(p[0]), float(p[1]))
        elif hasattr(track, "pos"):
            p = getattr(track, "pos")
            pos = (float(p[0]), float(p[1]))
        elif isinstance(track, dict):
            if "predicted_pos" in track:
                p = track["predicted_pos"]
                pos = (float(p[0]), float(p[1]))
            elif "pos" in track:
                p = track["pos"]
                pos = (float(p[0]), float(p[1]))
        elif isinstance(track, (list, tuple)) and len(track) >= 2:
            pos = (float(track[0]), float(track[1]))

        if hasattr(track, "kf") and hasattr(track.kf, "covariance"):
            cov = np.asarray(track.kf.covariance, dtype=np.float64)
            if hasattr(track.kf, "state") and track.kf.state.shape[0] >= 4:
                vel = (float(track.kf.state[2, 0]), float(track.kf.state[3, 0]))
        elif hasattr(track, "covariance"):
            cov = np.asarray(getattr(track, "covariance"), dtype=np.float64)
        elif hasattr(track, "state_covariance"):
            cov = np.asarray(getattr(track, "state_covariance"), dtype=np.float64)
        elif isinstance(track, dict) and "covariance" in track:
            cov = np.asarray(track["covariance"], dtype=np.float64)

        if hasattr(track, "velocity"):
            v = getattr(track, "velocity")
            vel = (float(v[0]), float(v[1]))
        elif isinstance(track, dict) and "velocity" in track:
            v = track["velocity"]
            vel = (float(v[0]), float(v[1]))

        if hasattr(track, "motion_model") and track.motion_model is not None:
            gate = float(track.motion_model.chi2_gate_threshold)
        elif isinstance(track, dict) and "chi2_gate_threshold" in track:
            gate = float(track["chi2_gate_threshold"])

        if cov.ndim == 1:
            cov = np.diag(cov)
        if cov.shape == (2, 2):
            full_cov = np.zeros((4, 4), dtype=np.float64)
            full_cov[:2, :2] = cov
            cov = full_cov

        return pos, cov, vel, gate

    def _extract_detection_info(
        self,
        det: Any,
        default_measurement_noise_std: float,
        time_uncertainty_seconds: float,
    ) -> tuple[tuple[float, float], np.ndarray, float]:
        pos = (0.0, 0.0)
        meas_cov = None
        sigma_t = float(time_uncertainty_seconds)

        if isinstance(det, dict):
            if "xy" in det:
                p = det["xy"]
                pos = (float(p[0]), float(p[1]))
            elif "pos" in det:
                p = det["pos"]
                pos = (float(p[0]), float(p[1]))
            if "measurement_covariance" in det and det["measurement_covariance"] is not None:
                meas_cov = np.asarray(det["measurement_covariance"], dtype=np.float64)
            if "time_uncertainty_seconds" in det and det["time_uncertainty_seconds"] is not None:
                sigma_t = float(det["time_uncertainty_seconds"])
        elif hasattr(det, "pos"):
            p = getattr(det, "pos")
            pos = (float(p[0]), float(p[1]))
        elif isinstance(det, (list, tuple)) and len(det) >= 2:
            pos = (float(det[0]), float(det[1]))

        if hasattr(det, "measurement_covariance") and getattr(det, "measurement_covariance") is not None:
            meas_cov = np.asarray(getattr(det, "measurement_covariance"), dtype=np.float64)

        if hasattr(det, "time_uncertainty_seconds") and getattr(det, "time_uncertainty_seconds") is not None:
            sigma_t = float(getattr(det, "time_uncertainty_seconds"))

        if meas_cov is None:
            meas_cov = np.eye(2, dtype=np.float64) * (default_measurement_noise_std**2)
        elif meas_cov.ndim == 1:
            meas_cov = np.diag(meas_cov)

        return pos, meas_cov, sigma_t

    def compute_innovation_covariance(
        self,
        p_pred: np.ndarray,
        r_meas: np.ndarray,
        velocity: tuple[float, float],
        sigma_t: float,
    ) -> np.ndarray:
        h_p_ht = p_pred[:2, :2]
        vx, vy = velocity
        if sigma_t > 0.0 and (vx != 0.0 or vy != 0.0):
            v_vec = np.array([[vx], [vy]], dtype=np.float64)
            r_time = (sigma_t**2) * (v_vec @ v_vec.T)
        else:
            r_time = np.zeros((2, 2), dtype=np.float64)
        return h_p_ht + r_meas + r_time

    def compute_mahalanobis_distance_sq(
        self,
        residual: np.ndarray,
        s_cov: np.ndarray,
    ) -> float:
        try:
            s_inv = np.linalg.inv(s_cov)
        except np.linalg.LinAlgError:
            s_inv = np.linalg.pinv(s_cov)
        val = residual.T @ s_inv @ residual
        return float(val.item())

    def associate(
        self,
        tracks: Sequence[Any],
        detections: Sequence[Any],
        gate_distance_m: Optional[float] = None,
        default_measurement_noise_std: float = 10.0,
        time_uncertainty_seconds: float = 0.0,
        gate_threshold_override: Optional[float] = None,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        num_tracks = len(tracks)
        num_detections = len(detections)

        if num_tracks == 0:
            return [], [], list(range(num_detections))
        if num_detections == 0:
            return [], list(range(num_tracks)), []

        cost_matrix = np.full((num_tracks, num_detections), np.inf, dtype=np.float64)

        for i, track in enumerate(tracks):
            pos_t, cov_t, vel_t, class_gate = self._extract_track_state(track)
            gate_val = gate_threshold_override if gate_threshold_override is not None else class_gate

            for j, det in enumerate(detections):
                pos_d, meas_cov, sigma_t = self._extract_detection_info(
                    det,
                    default_measurement_noise_std=default_measurement_noise_std,
                    time_uncertainty_seconds=time_uncertainty_seconds,
                )

                dx = pos_d[0] - pos_t[0]
                dy = pos_d[1] - pos_t[1]
                euc_dist = float(np.hypot(dx, dy))
                if gate_distance_m is not None and euc_dist > gate_distance_m:
                    continue

                s_cov = self.compute_innovation_covariance(
                    p_pred=cov_t,
                    r_meas=meas_cov,
                    velocity=vel_t,
                    sigma_t=sigma_t,
                )
                residual = np.array([[dx], [dy]], dtype=np.float64)
                d2 = self.compute_mahalanobis_distance_sq(residual, s_cov)

                if d2 <= gate_val:
                    cost_matrix[i, j] = d2

        solvable_cost = np.where(np.isinf(cost_matrix), 1e9, cost_matrix)
        row_ind, col_ind = linear_sum_assignment(solvable_cost)

        matched_pairs: list[tuple[int, int]] = []
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1e8:
                matched_pairs.append((int(r), int(c)))
                matched_tracks.add(int(r))
                matched_detections.add(int(c))

        unmatched_tracks = [i for i in range(num_tracks) if i not in matched_tracks]
        unmatched_detections = [j for j in range(num_detections) if j not in matched_detections]

        return matched_pairs, unmatched_tracks, unmatched_detections
