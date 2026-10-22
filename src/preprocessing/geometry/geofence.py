from typing import Tuple
import pyproj
import shapely
from shapely.geometry import Point, Polygon
from src.domain.enums import Capability, Zone


class GeofenceManager:
    capability: Capability = Capability.IMPLEMENTED

    def __init__(
        self,
        center_lat: float = 16.0,
        center_lon: float = 108.0,
        zone1_radius_km: float = 500.0,
        zone2_radius_km: float = 1500.0,
        zone3_radius_km: float = 3000.0,
    ) -> None:
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zone1_radius_m = zone1_radius_km * 1000.0
        self.zone2_radius_m = zone2_radius_km * 1000.0
        self.zone3_radius_m = zone3_radius_km * 1000.0

        self.geod = pyproj.Geod(ellps="WGS84")
        self.aeqd_proj_string = (
            f"+proj=aeqd +lat_0={self.center_lat} +lon_0={self.center_lon} +datum=WGS84 +units=m +no_defs"
        )
        self.proj_aeqd = pyproj.CRS(self.aeqd_proj_string)
        self.proj_wgs84 = pyproj.CRS("EPSG:4326")

        self.forward_transform = pyproj.Transformer.from_crs(
            self.proj_wgs84, self.proj_aeqd, always_xy=True
        ).transform
        self.backward_transform = pyproj.Transformer.from_crs(
            self.proj_aeqd, self.proj_wgs84, always_xy=True
        ).transform

    def compute_geodesic_distance_m(self, lat: float, lon: float) -> float:
        _, _, distance_m = self.geod.inv(self.center_lon, self.center_lat, lon, lat)
        return float(distance_m)

    def classify_zone(self, lat: float, lon: float) -> Zone:
        dist_m = self.compute_geodesic_distance_m(lat, lon)
        if dist_m <= self.zone1_radius_m:
            return Zone.ZONE_1
        if dist_m <= self.zone2_radius_m:
            return Zone.ZONE_2
        if dist_m <= self.zone3_radius_m:
            return Zone.ZONE_3
        return Zone.OUTSIDE_AOI

    def get_zone_buffer_polygon(self, radius_m: float, quad_segs: int = 90) -> Polygon:
        center_aeqd = Point(0.0, 0.0)
        circle_aeqd = center_aeqd.buffer(radius_m, quad_segs=quad_segs)
        polygon_wgs84 = shapely.transform(circle_aeqd, self.backward_transform, interleaved=False)
        return polygon_wgs84
