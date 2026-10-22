import math
from src.core.geofence import GeofenceManager


def test_center_distance():
    manager = GeofenceManager()
    dist = manager.compute_geodesic_distance_m(16.0, 108.0)
    assert dist == 0.0
    assert manager.classify_zone(16.0, 108.0) == 1


def test_zone_classification():
    manager = GeofenceManager()
    assert manager.classify_zone(16.5, 108.5) == 1

    dist_1000km_pt = (16.0, 117.0)
    d = manager.compute_geodesic_distance_m(dist_1000km_pt[0], dist_1000km_pt[1])
    assert 500_000.0 < d < 1_500_000.0
    assert manager.classify_zone(dist_1000km_pt[0], dist_1000km_pt[1]) == 2

    dist_2000km_pt = (16.0, 126.0)
    d_2000 = manager.compute_geodesic_distance_m(dist_2000km_pt[0], dist_2000km_pt[1])
    assert 1_500_000.0 < d_2000 < 3_000_000.0
    assert manager.classify_zone(dist_2000km_pt[0], dist_2000km_pt[1]) == 3

    dist_far = (45.0, 108.0)
    assert manager.classify_zone(dist_far[0], dist_far[1]) == 0


def test_aeqd_area_consistency():
    manager = GeofenceManager()
    radius_m = 500_000.0
    poly = manager.get_zone_buffer_polygon(radius_m, quad_segs=180)
    poly_aeqd = manager.forward_transform(poly.exterior.coords.xy[0], poly.exterior.coords.xy[1])
    from shapely.geometry import Polygon
    transformed_poly = Polygon(zip(poly_aeqd[0], poly_aeqd[1]))
    expected_area = math.pi * (radius_m**2)
    relative_error = abs(transformed_poly.area - expected_area) / expected_area
    assert relative_error < 0.001
