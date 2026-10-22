from typing import Any, Sequence
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
