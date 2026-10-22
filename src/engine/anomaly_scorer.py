from typing import Dict, Optional


class AnomalyScorer:
    def __init__(self, weights: Dict[str, float], reference_distance_km: float = 500.0) -> None:
        self.weights = weights
        self.reference_distance_km = reference_distance_km
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    def score_zone(self, zone: int) -> float:
        if zone == 1:
            return 1.0
        if zone == 2:
            return 0.6
        if zone == 3:
            return 0.2
        return 0.0

    def score_dark_status(self, is_dark: Optional[bool]) -> float:
        if is_dark is True:
            return 1.0
        if is_dark is None:
            return 0.5
        return 0.0

    def score_boundary_proximity(self, geodesic_distance_km: float) -> float:
        if self.reference_distance_km <= 0:
            return 0.0
        val = 1.0 - (geodesic_distance_km / self.reference_distance_km)
        return max(0.0, min(1.0, val))

    def calculate_score(
        self,
        zone: int,
        is_dark: Optional[bool],
        velocity_score: float = 0.0,
        change_score: float = 0.0,
        geodesic_distance_km: float = 500.0,
    ) -> float:
        s_z = self.score_zone(zone)
        s_d = self.score_dark_status(is_dark)
        s_v = max(0.0, min(1.0, velocity_score))
        s_c = max(0.0, min(1.0, change_score))
        s_b = self.score_boundary_proximity(geodesic_distance_km)

        raw_score = (
            self.weights["w_z"] * s_z
            + self.weights["w_d"] * s_d
            + self.weights["w_v"] * s_v
            + self.weights["w_c"] * s_c
            + self.weights["w_b"] * s_b
        )
        final_s = 100.0 * raw_score
        return max(0.0, min(100.0, final_s))
