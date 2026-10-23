from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple
import numpy as np


@dataclass(frozen=True)
class AisRecord:
    mmsi: str
    lat: float
    lon: float
    timestamp: datetime
    vessel_name: str
    vessel_type: str
    speed_knots: float
    heading_deg: float


def generate_synthetic_sar_scene(
    height: int = 100,
    width: int = 100,
    background_mean: float = 10.0,
    targets: List[Tuple[int, int, int, int, float]] = None,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scene = rng.exponential(scale=background_mean, size=(height, width)).astype(np.float32)

    if targets is None:
        targets = [
            (30, 30, 5, 2, 400.0),
            (70, 70, 3, 6, 450.0),
        ]

    for r_start, c_start, r_len, c_len, val in targets:
        scene[r_start : r_start + r_len, c_start : c_start + c_len] = val

    return scene


def generate_mock_ais_records(
    base_time: datetime = None,
) -> List[AisRecord]:
    if base_time is None:
        base_time = datetime.now(timezone.utc)

    return [
        AisRecord(
            mmsi="412123456",
            lat=16.03,
            lon=108.03,
            timestamp=base_time,
            vessel_name="PACIFIC_NAVIGATOR",
            vessel_type="CARGO",
            speed_knots=12.5,
            heading_deg=45.0,
        ),
        AisRecord(
            mmsi="412987654",
            lat=16.20,
            lon=108.20,
            timestamp=base_time,
            vessel_name="EAST_SEA_FISHER",
            vessel_type="FISHING",
            speed_knots=4.0,
            heading_deg=180.0,
        ),
    ]


def generate_synthetic_optical_airport(
    height: int = 120,
    width: int = 120,
    aircraft_positions: List[Tuple[int, int, int, int]] = None,
    background_val: float = 0.2,
    aircraft_val: float = 0.9,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((3, height, width), background_val, dtype=np.float32)
    noise = rng.normal(0.0, 0.02, size=(3, height, width)).astype(np.float32)
    image = np.clip(image + noise, 0.0, 1.0)

    if aircraft_positions is None:
        aircraft_positions = [
            (30, 40, 12, 6),
            (75, 80, 14, 8),
        ]

    for r, c, l, w in aircraft_positions:
        image[:, r : r + l, c : c + w] = aircraft_val

    return image


def generate_synthetic_parallax_scene(
    height: int = 100,
    width: int = 100,
    triplets: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None,
    background_val: float = 0.2,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scene = np.full((3, height, width), background_val, dtype=np.float32)
    noise = rng.normal(0.0, 0.01, size=(3, height, width)).astype(np.float32)
    scene = np.clip(scene + noise, 0.0, 1.0)

    if triplets is None:
        triplets = [
            ((40, 40), (43, 44), (46, 48)),
        ]

    for red_pt, grn_pt, blu_pt in triplets:
        scene[0, red_pt[0] - 1 : red_pt[0] + 2, red_pt[1] - 1 : red_pt[1] + 2] = 0.95
        scene[1, grn_pt[0] - 1 : grn_pt[0] + 2, grn_pt[1] - 1 : grn_pt[1] + 2] = 0.95
        scene[2, blu_pt[0] - 1 : blu_pt[0] + 2, blu_pt[1] - 1 : blu_pt[1] + 2] = 0.95

    return scene


def generate_synthetic_bitemporal_pair(
    height: int = 100,
    width: int = 100,
    new_structures: List[Tuple[int, int, int, int, float]] = None,
    background_val: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t1 = np.full((height, width), background_val, dtype=np.float32)
    t1 += rng.normal(0.0, 0.02, size=(height, width)).astype(np.float32)
    t1 = np.clip(t1, 0.0, 1.0)

    t2 = t1.copy()
    if new_structures is None:
        new_structures = [
            (45, 45, 12, 12, 0.85),
            (70, 20, 8, 8, 0.80),
        ]

    for r, c, h_len, w_len, val in new_structures:
        t2[r : r + h_len, c : c + w_len] = val

    return t1, t2
