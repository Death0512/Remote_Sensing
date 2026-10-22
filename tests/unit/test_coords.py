import pytest
from src.core.coords import AffineCoordinateTransformer


def test_round_trip_accuracy():
    transform = (0.0001, 0.0, 108.0, 0.0, -0.0001, 16.0)
    transformer = AffineCoordinateTransformer(transform)

    test_pixels = [(0.0, 0.0), (100.5, 200.25), (799.0, 799.0), (1234.56, 789.12)]
    for col, row in test_pixels:
        x, y = transformer.pixel_to_geo(col, row)
        rec_col, rec_row = transformer.geo_to_pixel(x, y)
        assert abs(rec_col - col) < 1e-4
        assert abs(rec_row - row) < 1e-4


def test_singular_matrix():
    transform = (1.0, 2.0, 0.0, 2.0, 4.0, 0.0)
    with pytest.raises(ValueError):
        AffineCoordinateTransformer(transform)
