import math
from typing import Generator, List, Tuple


class TileWindow:
    def __init__(self, col_off: int, row_off: int, width: int, height: int) -> None:
        self.col_off = col_off
        self.row_off = row_off
        self.width = width
        self.height = height


def generate_tiling_windows(
    image_width: int, image_height: int, tile_size: int = 800, overlap_px: int = 128
) -> Generator[TileWindow, None, None]:
    step = tile_size - overlap_px
    if step <= 0:
        raise ValueError("overlap_px must be strictly smaller than tile_size")

    row_off = 0
    while row_off < image_height:
        actual_height = min(tile_size, image_height - row_off)
        col_off = 0
        while col_off < image_width:
            actual_width = min(tile_size, image_width - col_off)
            yield TileWindow(col_off, row_off, actual_width, actual_height)
            if col_off + tile_size >= image_width:
                break
            col_off += step
        if row_off + tile_size >= image_height:
            break
        row_off += step


def point_deduplicate(
    points: List[Tuple[float, float, float]], distance_threshold_px: float = 20.0
) -> List[Tuple[float, float, float]]:
    sorted_pts = sorted(points, key=lambda p: p[2], reverse=True)
    retained: List[Tuple[float, float, float]] = []

    for x, y, score in sorted_pts:
        is_duplicate = False
        for rx, ry, _ in retained:
            dist = math.hypot(x - rx, y - ry)
            if dist < distance_threshold_px:
                is_duplicate = True
                break
        if not is_duplicate:
            retained.append((x, y, score))

    return retained
