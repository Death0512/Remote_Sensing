import numpy as np


class KalmanFilter2D:
    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray | None = None,
        process_noise_std: float = 1.0,
        measurement_noise_std: float = 5.0,
    ) -> None:
        self.state = np.asarray(initial_state, dtype=np.float64).reshape(4, 1)
        if initial_covariance is not None:
            self.covariance = np.asarray(initial_covariance, dtype=np.float64).reshape(4, 4)
        else:
            self.covariance = np.diag([measurement_noise_std**2, measurement_noise_std**2, 100.0, 100.0])
        self.process_noise_std = float(process_noise_std)
        self.measurement_noise_std = float(measurement_noise_std)
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.R = np.eye(2, dtype=np.float64) * (self.measurement_noise_std**2)

    def predict(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        dt = float(dt)
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
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
        return self.state.copy(), self.covariance.copy()

    def update(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = np.asarray(measurement, dtype=np.float64).reshape(2, 1)
        y = z - (self.H @ self.state)
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + (K @ y)
        I = np.eye(4, dtype=np.float64)
        I_KH = I - (K @ self.H)
        self.covariance = I_KH @ self.covariance @ I_KH.T + K @ self.R @ K.T
        return self.state.copy(), self.covariance.copy()
