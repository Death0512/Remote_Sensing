from dataclasses import dataclass
import math
from typing import Any, Optional
from src.domain.enums import GeometryStatus


@dataclass(frozen=True)
class AxisAlignedBoundingBox:
    min_col: float
    min_row: float
    max_col: float
    max_row: float

    def __post_init__(self) -> None:
        min_c = float(self.min_col)
        max_c = float(self.max_col)
        min_r = float(self.min_row)
        max_r = float(self.max_row)
        if min_c > max_c:
            min_c, max_c = max_c, min_c
        if min_r > max_r:
            min_r, max_r = max_r, min_r
        object.__setattr__(self, "min_col", min_c)
        object.__setattr__(self, "min_row", min_r)
        object.__setattr__(self, "max_col", max_c)
        object.__setattr__(self, "max_row", max_r)

    @property
    def width(self) -> float:
        return self.max_col - self.min_col

    @property
    def height(self) -> float:
        return self.max_row - self.min_row

    @property
    def cx(self) -> float:
        return (self.min_col + self.max_col) / 2.0

    @property
    def cy(self) -> float:
        return (self.min_row + self.max_row) / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0.0 else 1.0

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.cx, self.cy)

    @property
    def corners(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
        return (
            (self.min_col, self.min_row),
            (self.max_col, self.min_row),
            (self.max_col, self.max_row),
            (self.min_col, self.max_row),
        )

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.min_col, self.min_row, self.max_col, self.max_row)

    def to_obb(self) -> "OrientedBoundingBox":
        return OrientedBoundingBox(
            cx=self.cx,
            cy=self.cy,
            width=self.width,
            height=self.height,
            angle_deg=0.0,
        )

    def to_polygon(self) -> Any:
        from shapely.geometry import Polygon

        return Polygon(self.corners)

    def __iter__(self):
        return iter(self.to_tuple())

    def __getitem__(self, idx: int) -> float:
        return self.to_tuple()[idx]

    def __len__(self) -> int:
        return 4

    @classmethod
    def from_tuple(cls, coords: tuple[float, float, float, float]) -> "AxisAlignedBoundingBox":
        return cls(
            min_col=float(coords[0]),
            min_row=float(coords[1]),
            max_col=float(coords[2]),
            max_row=float(coords[3]),
        )


@dataclass(frozen=True)
class OrientedBoundingBox:
    cx: float
    cy: float
    width: float
    height: float
    angle_deg: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "cx", float(self.cx))
        object.__setattr__(self, "cy", float(self.cy))
        object.__setattr__(self, "width", max(0.0, float(self.width)))
        object.__setattr__(self, "height", max(0.0, float(self.height)))
        object.__setattr__(self, "angle_deg", float(self.angle_deg) % 360.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0.0 else 1.0

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.cx, self.cy)

    @property
    def corners(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
        rad = math.radians(self.angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        w2 = self.width / 2.0
        h2 = self.height / 2.0
        offsets = ((-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2))
        return tuple(
            (self.cx + x * cos_a - y * sin_a, self.cy + x * sin_a + y * cos_a)
            for x, y in offsets
        )

    def to_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.cx, self.cy, self.width, self.height, self.angle_deg)

    def to_aabb(self) -> AxisAlignedBoundingBox:
        c = self.corners
        cols = [pt[0] for pt in c]
        rows = [pt[1] for pt in c]
        return AxisAlignedBoundingBox(
            min_col=min(cols),
            min_row=min(rows),
            max_col=max(cols),
            max_row=max(rows),
        )

    def to_polygon(self) -> Any:
        from shapely.geometry import Polygon

        return Polygon(self.corners)

    def __iter__(self):
        return iter(self.to_tuple())

    def __getitem__(self, idx: int) -> float:
        return self.to_tuple()[idx]

    def __len__(self) -> int:
        return 5

    @classmethod
    def from_tuple(cls, coords: tuple[float, float, float, float, float]) -> "OrientedBoundingBox":
        return cls(
            cx=float(coords[0]),
            cy=float(coords[1]),
            width=float(coords[2]),
            height=float(coords[3]),
            angle_deg=float(coords[4]),
        )

    @classmethod
    def from_xywhr(
        cls,
        cx: float,
        cy: float,
        width: float,
        height: float,
        angle_rad: float,
    ) -> "OrientedBoundingBox":
        return cls(
            cx=float(cx),
            cy=float(cy),
            width=float(width),
            height=float(height),
            angle_deg=float(math.degrees(angle_rad)),
        )


@dataclass(frozen=True)
class PixelGeometry:
    row: float
    col: float
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple):
            object.__setattr__(self, "shape", tuple(self.shape))


@dataclass(frozen=True)
class MapGeometry:
    x: float
    y: float
    crs: str
    affine: tuple[float, float, float, float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.affine, tuple):
            object.__setattr__(self, "affine", tuple(self.affine))


@dataclass(frozen=True)
class GeodeticGeometry:
    lon: float
    lat: float
    crs: str = "EPSG:4326"
    horizontal_accuracy_m: Optional[float] = None


__all__ = [
    "AxisAlignedBoundingBox",
    "GeodeticGeometry",
    "GeometryStatus",
    "MapGeometry",
    "OrientedBoundingBox",
    "PixelGeometry",
]
