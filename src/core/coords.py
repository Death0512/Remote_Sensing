from typing import Tuple


class AffineCoordinateTransformer:
    def __init__(self, transform_matrix: Tuple[float, float, float, float, float, float]) -> None:
        self.a, self.b, self.c, self.d, self.e, self.f = transform_matrix
        self.det = self.a * self.e - self.b * self.d
        if abs(self.det) < 1e-12:
            raise ValueError("Singular affine transform matrix.")

    def pixel_to_geo(self, col: float, row: float) -> Tuple[float, float]:
        x = self.c + self.a * col + self.b * row
        y = self.f + self.d * col + self.e * row
        return x, y

    def geo_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        dx = x - self.c
        dy = y - self.f
        col = (self.e * dx - self.b * dy) / self.det
        row = (-self.d * dx + self.a * dy) / self.det
        return col, row
