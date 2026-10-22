from dataclasses import dataclass
import math
from typing import Generator, List, Tuple
from src.domain.enums import Capability


@dataclass(frozen=True)
class MetricTileWindow:
    col_off: int
    row_off: int
    width: int
    height: int
    col_slice: slice
    row_slice: slice
    window_m: float
    gsd_m: float


class MetricTiler:
    capability: Capability = Capability.IMPLEMENTED

    def __init__(
        self,
        metric_window_m: float = 8000.0,
        metric_overlap_m: float = 1000.0,
        gsd_m: float = 10.0,
    ) -> None:
        if gsd_m <= 0:
            raise ValueError("gsd_m must be strictly positive")
        if metric_window_m <= 0:
            raise ValueError("metric_window_m must be strictly positive")
        if metric_overlap_m < 0:
            raise ValueError("metric_overlap_m must be non-negative")
        if metric_overlap_m >= metric_window_m:
            raise ValueError("metric_overlap_m must be strictly smaller than metric_window_m")

        self.metric_window_m = metric_window_m
        self.metric_overlap_m = metric_overlap_m
        self.gsd_m = gsd_m

        self.window_px = max(1, int(round(metric_window_m / gsd_m)))
        self.overlap_px = int(round(metric_overlap_m / gsd_m))
        self.step_px = self.window_px - self.overlap_px
        if self.step_px <= 0:
            raise ValueError("Tile step in pixels must be strictly positive")

    def generate_windows(
        self, image_width: int, image_height: int
    ) -> Generator[MetricTileWindow, None, None]:
        if image_width <= 0 or image_height <= 0:
            return

        row_off = 0
        while row_off < image_height:
            actual_height = min(self.window_px, image_height - row_off)
            col_off = 0
            while col_off < image_width:
                actual_width = min(self.window_px, image_width - col_off)
                yield MetricTileWindow(
                    col_off=col_off,
                    row_off=row_off,
                    width=actual_width,
                    height=actual_height,
                    col_slice=slice(col_off, col_off + actual_width),
                    row_slice=slice(row_off, row_off + actual_height),
                    window_m=self.metric_window_m,
                    gsd_m=self.gsd_m,
                )
                if col_off + self.window_px >= image_width:
                    break
                col_off += self.step_px
            if row_off + self.window_px >= image_height:
                break
            row_off += self.step_px


def generate_metric_tiling_windows(
    image_width: int,
    image_height: int,
    metric_window_m: float,
    metric_overlap_m: float,
    gsd_m: float,
) -> Generator[MetricTileWindow, None, None]:
    tiler = MetricTiler(
        metric_window_m=metric_window_m,
        metric_overlap_m=metric_overlap_m,
        gsd_m=gsd_m,
    )
    yield from tiler.generate_windows(image_width, image_height)


def point_deduplicate_metric(
    points: List[Tuple[float, float, float]],
    metric_dedup_m: float,
    gsd_m: float,
) -> List[Tuple[float, float, float]]:
    if gsd_m <= 0:
        raise ValueError("gsd_m must be strictly positive")
    if metric_dedup_m < 0:
        raise ValueError("metric_dedup_m must be non-negative")

    threshold_px = metric_dedup_m / gsd_m
    sorted_pts = sorted(points, key=lambda p: p[2], reverse=True)
    retained: List[Tuple[float, float, float]] = []

    for x, y, score in sorted_pts:
        is_duplicate = False
        for rx, ry, _ in retained:
            dist = math.hypot(x - rx, y - ry)
            if dist < threshold_px:
                is_duplicate = True
                break
        if not is_duplicate:
            retained.append((x, y, score))

    return retained
