from abc import ABC, abstractmethod
from typing import Optional, Sequence
import math
import numpy as np

from src.domain.enums import CalibrationState


class ProbabilityCalibrator(ABC):
    @property
    @abstractmethod
    def calibration_state(self) -> CalibrationState:
        pass

    @abstractmethod
    def calibrate(self, score: float) -> float:
        pass


class PlattCalibrator(ProbabilityCalibrator):
    def __init__(self, a: float = -1.0, b: float = 0.0) -> None:
        self.a = float(a)
        self.b = float(b)

    @property
    def calibration_state(self) -> CalibrationState:
        return CalibrationState.PLATT

    def fit(
        self,
        scores: Sequence[float],
        labels: Sequence[int],
        lr: float = 0.05,
        epochs: int = 500,
    ) -> "PlattCalibrator":
        s = np.asarray(scores, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if len(s) == 0 or len(s) != len(y):
            raise ValueError("Scores and labels must be non-empty and equal length")

        a = -1.0
        b = 0.0
        n = len(s)
        for _ in range(epochs):
            z = a * s + b
            z_clipped = np.clip(z, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-z_clipped))
            grad_a = np.sum((p - y) * s) / n
            grad_b = np.sum(p - y) / n
            a -= lr * float(grad_a)
            b -= lr * float(grad_b)

        self.a = a
        self.b = b
        return self

    def calibrate(self, score: float) -> float:
        z = self.a * float(score) + self.b
        z = max(min(z, 30.0), -30.0)
        p = 1.0 / (1.0 + math.exp(-z))
        return float(np.clip(p, 0.0, 1.0))


class IsotonicCalibrator(ProbabilityCalibrator):
    def __init__(
        self,
        thresholds: Optional[Sequence[float]] = None,
        calibrated_values: Optional[Sequence[float]] = None,
    ) -> None:
        self.thresholds = list(thresholds) if thresholds is not None else []
        self.calibrated_values = (
            list(calibrated_values) if calibrated_values is not None else []
        )

    @property
    def calibration_state(self) -> CalibrationState:
        return CalibrationState.ISOTONIC

    def fit(
        self,
        scores: Sequence[float],
        labels: Sequence[int],
    ) -> "IsotonicCalibrator":
        s = np.asarray(scores, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if len(s) == 0 or len(s) != len(y):
            raise ValueError("Scores and labels must be non-empty and equal length")

        order = np.argsort(s)
        s_sorted = s[order]
        y_sorted = y[order]

        blocks = [[y_sorted[i], 1, [s_sorted[i]]] for i in range(len(s_sorted))]
        i = 0
        while i < len(blocks) - 1:
            if blocks[i][0] > blocks[i + 1][0]:
                w1 = blocks[i][1]
                w2 = blocks[i + 1][1]
                val = (blocks[i][0] * w1 + blocks[i + 1][0] * w2) / (w1 + w2)
                blocks[i][0] = val
                blocks[i][1] = w1 + w2
                blocks[i][2].extend(blocks[i + 1][2])
                del blocks[i + 1]
                if i > 0:
                    i -= 1
            else:
                i += 1

        self.thresholds = []
        self.calibrated_values = []
        for block in blocks:
            val = float(np.clip(block[0], 0.0, 1.0))
            max_s = float(max(block[2]))
            self.thresholds.append(max_s)
            self.calibrated_values.append(val)
        return self

    def calibrate(self, score: float) -> float:
        if not self.thresholds:
            return float(np.clip(score, 0.0, 1.0))
        s = float(score)
        if s <= self.thresholds[0]:
            return self.calibrated_values[0]
        if s >= self.thresholds[-1]:
            return self.calibrated_values[-1]
        for idx, th in enumerate(self.thresholds):
            if s <= th:
                return self.calibrated_values[idx]
        return self.calibrated_values[-1]


class ReliabilityCurveCalibrator(ProbabilityCalibrator):
    def __init__(
        self,
        num_bins: int = 10,
        bin_edges: Optional[Sequence[float]] = None,
        bin_probabilities: Optional[Sequence[float]] = None,
    ) -> None:
        self.num_bins = num_bins
        self.bin_edges = (
            list(bin_edges)
            if bin_edges is not None
            else [i / num_bins for i in range(num_bins + 1)]
        )
        self.bin_probabilities = (
            list(bin_probabilities) if bin_probabilities is not None else []
        )

    @property
    def calibration_state(self) -> CalibrationState:
        return CalibrationState.RELIABILITY_CURVE

    def fit(
        self,
        scores: Sequence[float],
        labels: Sequence[int],
    ) -> "ReliabilityCurveCalibrator":
        s = np.asarray(scores, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if len(s) == 0 or len(s) != len(y):
            raise ValueError("Scores and labels must be non-empty and equal length")

        edges = np.linspace(0.0, 1.0, self.num_bins + 1)
        self.bin_edges = [float(e) for e in edges]
        bin_probs = []

        for b in range(self.num_bins):
            low = edges[b]
            high = edges[b + 1]
            if b == self.num_bins - 1:
                mask = (s >= low) & (s <= high)
            else:
                mask = (s >= low) & (s < high)

            if np.any(mask):
                prob = float(np.mean(y[mask]))
            else:
                prob = float(0.5 * (low + high))
            bin_probs.append(float(np.clip(prob, 0.0, 1.0)))

        self.bin_probabilities = bin_probs
        return self

    def calibrate(self, score: float) -> float:
        if not self.bin_probabilities:
            return float(np.clip(score, 0.0, 1.0))
        s = float(score)
        if s <= self.bin_edges[0]:
            return self.bin_probabilities[0]
        if s >= self.bin_edges[-1]:
            return self.bin_probabilities[-1]

        for b in range(len(self.bin_probabilities)):
            if s < self.bin_edges[b + 1] or (
                b == len(self.bin_probabilities) - 1 and s <= self.bin_edges[b + 1]
            ):
                return self.bin_probabilities[b]
        return self.bin_probabilities[-1]
