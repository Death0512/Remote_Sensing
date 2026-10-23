import math
from typing import Optional
import numpy as np
from src.domain.enums import MotionClass
from src.tracking.motion_models import MotionModelConfig


class KalmanFilter2D:
    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: Optional[np.ndarray] = None,
        process_noise_std: float = 1.0,
        measurement_noise_std: float = 5.0,
        q_pos: Optional[float] = None,
        q_vel: Optional[float] = None,
        motion_model: Optional[MotionModelConfig] = None,
    ) -> None:
        self.state = np.asarray(initial_state, dtype=np.float64).reshape(4, 1)
        self.motion_model = motion_model
        meas_std = (
            float(motion_model.default_measurement_noise_std)
            if motion_model is not None
            else float(measurement_noise_std)
        )
        if initial_covariance is not None:
            self.covariance = np.asarray(initial_covariance, dtype=np.float64).reshape(4, 4)
        else:
            self.covariance = np.diag([meas_std**2, meas_std**2, 100.0, 100.0])
        self.process_noise_std = float(process_noise_std)
        self.measurement_noise_std = meas_std
        if motion_model is not None:
            self.q_pos: Optional[float] = float(motion_model.process_noise_pos)
            self.q_vel: Optional[float] = float(motion_model.process_noise_vel)
        else:
            self.q_pos = float(q_pos) if q_pos is not None else None
            self.q_vel = float(q_vel) if q_vel is not None else None
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.R = np.eye(2, dtype=np.float64) * (self.measurement_noise_std**2)
        self._enforce_kinematic_bounds()

    def _enforce_kinematic_bounds(self) -> None:
        if self.motion_model is None:
            return
        if self.motion_model.motion_class == MotionClass.STATIONARY:
            self.state[2, 0] = 0.0
            self.state[3, 0] = 0.0
            return
        vx = float(self.state[2, 0])
        vy = float(self.state[3, 0])
        speed = math.hypot(vx, vy)
        max_speed = self.motion_model.max_speed_mps
        if speed > max_speed and speed > 1e-6:
            scale = max_speed / speed
            self.state[2, 0] = vx * scale
            self.state[3, 0] = vy * scale

    def predict(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        dt = float(dt)
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        if self.motion_model is not None and self.motion_model.motion_class == MotionClass.STATIONARY:
            F = np.eye(4, dtype=np.float64)
            q_p = self.q_pos**2 if self.q_pos is not None else 0.0
            Q = np.diag([q_p * dt, q_p * dt, 0.0, 0.0])
            self.state[2, 0] = 0.0
            self.state[3, 0] = 0.0
        else:
            F = np.array(
                [
                    [1.0, 0.0, dt, 0.0],
                    [0.0, 1.0, 0.0, dt],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            if self.q_pos is not None and self.q_vel is not None:
                q_v2 = self.q_vel**2
                q_p2 = self.q_pos**2
                dt2 = dt**2
                dt3 = dt**3 / 2.0
                dt4 = dt**4 / 4.0
                Q = np.array(
                    [
                        [dt4 * q_v2 + q_p2 * dt, 0.0, dt3 * q_v2, 0.0],
                        [0.0, dt4 * q_v2 + q_p2 * dt, 0.0, dt3 * q_v2],
                        [dt3 * q_v2, 0.0, dt2 * q_v2, 0.0],
                        [0.0, dt3 * q_v2, 0.0, dt2 * q_v2],
                    ],
                    dtype=np.float64,
                )
            else:
                q = self.process_noise_std**2
                dt2 = dt**2
                dt3 = dt**3 / 2.0
                dt4 = dt**4 / 4.0
                Q = np.array(
                    [
                        [dt4 * q, 0.0, dt3 * q, 0.0],
                        [0.0, dt4 * q, 0.0, dt3 * q],
                        [dt3 * q, 0.0, dt2 * q, 0.0],
                        [0.0, dt3 * q, 0.0, dt2 * q],
                    ],
                    dtype=np.float64,
                )
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q
        self._enforce_kinematic_bounds()
        return self.state.copy(), self.covariance.copy()

    def update(
        self,
        measurement: np.ndarray,
        measurement_cov: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        z = np.asarray(measurement, dtype=np.float64).reshape(2, 1)
        R = np.asarray(measurement_cov, dtype=np.float64) if measurement_cov is not None else self.R
        y = z - (self.H @ self.state)
        S = self.H @ self.covariance @ self.H.T + R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + (K @ y)
        I = np.eye(4, dtype=np.float64)
        I_KH = I - (K @ self.H)
        self.covariance = I_KH @ self.covariance @ I_KH.T + K @ R @ K.T
        self._enforce_kinematic_bounds()
        return self.state.copy(), self.covariance.copy()
