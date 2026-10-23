import math
from typing import Any, Optional, Tuple, Union
import pyproj
from pyproj import CRS, Transformer

from src.domain.enums import Capability, GeometryStatus
from src.domain.geometry import GeodeticGeometry, MapGeometry, PixelGeometry
from src.ingestion.exceptions import InvalidGeoreferenceError


class GeoreferenceValidationError(InvalidGeoreferenceError):
    pass


class GeoreferencingPipeline:
    capability: Capability = Capability.IMPLEMENTED

    def __init__(
        self,
        crs: Optional[Union[str, CRS]] = None,
        affine: Optional[Tuple[float, float, float, float, float, float]] = None,
        max_roundtrip_residual_px: float = 0.5,
    ) -> None:
        self.max_roundtrip_residual_px = max_roundtrip_residual_px
        self.crs: Optional[CRS] = None
        self.crs_str: Optional[str] = None
        self.affine: Optional[Tuple[float, float, float, float, float, float]] = None
        self.inv_affine: Optional[Tuple[float, float, float, float, float, float]] = None
        self.to_wgs84_transformer: Optional[Transformer] = None
        self.from_wgs84_transformer: Optional[Transformer] = None
        self.is_valid: bool = False

        if crs is not None and affine is not None:
            self.configure(crs=crs, affine=affine)

    def configure(
        self,
        crs: Union[str, CRS],
        affine: Any,
    ) -> None:
        try:
            if isinstance(crs, CRS):
                self.crs = crs
                self.crs_str = crs.to_string()
            else:
                self.crs = CRS.from_user_input(crs)
                self.crs_str = str(crs)
        except Exception as exc:
            self.is_valid = False
            raise GeoreferenceValidationError(f"Invalid CRS: {exc}") from exc

        if hasattr(affine, "to_gdal"):
            g = affine.to_gdal()
            aff = (float(g[1]), float(g[2]), float(g[0]), float(g[4]), float(g[5]), float(g[3]))
        elif isinstance(affine, (tuple, list)) and len(affine) == 6:
            aff = tuple(float(v) for v in affine)
        else:
            self.is_valid = False
            raise GeoreferenceValidationError(f"Affine geotransform must be 6 floats, got {type(affine)}")

        a, b, c, d, e, f = aff
        det = a * e - b * d
        if abs(det) < 1e-12:
            self.is_valid = False
            raise GeoreferenceValidationError(f"Affine determinant is degenerate: {det}")

        is_identity = (
            math.isclose(a, 1.0, abs_tol=1e-7)
            and math.isclose(b, 0.0, abs_tol=1e-7)
            and math.isclose(c, 0.0, abs_tol=1e-7)
            and math.isclose(d, 0.0, abs_tol=1e-7)
            and (math.isclose(e, 1.0, abs_tol=1e-7) or math.isclose(e, -1.0, abs_tol=1e-7))
            and math.isclose(f, 0.0, abs_tol=1e-7)
        )
        if is_identity:
            self.is_valid = False
            raise GeoreferenceValidationError("Affine geotransform is unreferenced identity")

        inv_det = 1.0 / det
        inv_a = e * inv_det
        inv_b = -b * inv_det
        inv_c = (b * f - c * e) * inv_det
        inv_d = -d * inv_det
        inv_e = a * inv_det
        inv_f = (c * d - a * f) * inv_det

        self.affine = aff
        self.inv_affine = (inv_a, inv_b, inv_c, inv_d, inv_e, inv_f)

        try:
            wgs84 = CRS.from_epsg(4326)
            self.to_wgs84_transformer = Transformer.from_crs(self.crs, wgs84, always_xy=True)
            self.from_wgs84_transformer = Transformer.from_crs(wgs84, self.crs, always_xy=True)
            self.is_valid = True
        except Exception as exc:
            self.is_valid = False
            raise GeoreferenceValidationError(f"Failed to create coordinate transformer: {exc}") from exc

    def pixel_to_map(self, pixel: PixelGeometry) -> MapGeometry:
        if not self.is_valid or self.affine is None or self.crs_str is None:
            raise GeoreferenceValidationError("Pipeline has invalid or unconfigured georeferencing")

        a, b, c, d, e, f = self.affine
        x = a * pixel.col + b * pixel.row + c
        y = d * pixel.col + e * pixel.row + f
        return MapGeometry(x=float(x), y=float(y), crs=self.crs_str, affine=self.affine)

    def map_to_pixel(self, map_geom: MapGeometry, shape: Tuple[int, int] = (0, 0)) -> PixelGeometry:
        if not self.is_valid or self.inv_affine is None:
            raise GeoreferenceValidationError("Pipeline has invalid or unconfigured georeferencing")

        ia, ib, ic, id_, ie, if_ = self.inv_affine
        col = ia * map_geom.x + ib * map_geom.y + ic
        row = id_ * map_geom.x + ie * map_geom.y + if_
        return PixelGeometry(row=float(row), col=float(col), shape=shape)

    def map_to_geodetic(self, map_geom: MapGeometry) -> GeodeticGeometry:
        if not self.is_valid or self.to_wgs84_transformer is None:
            raise GeoreferenceValidationError("Pipeline has invalid or unconfigured georeferencing")

        lon, lat = self.to_wgs84_transformer.transform(map_geom.x, map_geom.y)
        return GeodeticGeometry(lon=float(lon), lat=float(lat), crs="EPSG:4326")

    def geodetic_to_map(self, geodetic: GeodeticGeometry) -> MapGeometry:
        if not self.is_valid or self.from_wgs84_transformer is None or self.affine is None or self.crs_str is None:
            raise GeoreferenceValidationError("Pipeline has invalid or unconfigured georeferencing")

        x, y = self.from_wgs84_transformer.transform(geodetic.lon, geodetic.lat)
        return MapGeometry(x=float(x), y=float(y), crs=self.crs_str, affine=self.affine)

    def forward(
        self,
        pixel: PixelGeometry,
        validate_roundtrip: bool = True,
    ) -> Tuple[GeodeticGeometry, GeometryStatus]:
        if not self.is_valid:
            return (
                GeodeticGeometry(lon=float(pixel.col), lat=float(pixel.row), crs="EPSG:4326"),
                GeometryStatus.UNKNOWN,
            )

        map_geom = self.pixel_to_map(pixel)
        geodetic = self.map_to_geodetic(map_geom)

        if validate_roundtrip:
            res_px = self.evaluate_roundtrip_error(pixel, geodetic)
            if res_px >= self.max_roundtrip_residual_px:
                raise GeoreferenceValidationError(
                    f"Roundtrip transformation error {res_px:.4f}px exceeds limit {self.max_roundtrip_residual_px}px"
                )

        return geodetic, GeometryStatus.GEOCODED

    def backward(self, geodetic: GeodeticGeometry, shape: Tuple[int, int] = (0, 0)) -> PixelGeometry:
        if not self.is_valid:
            raise GeoreferenceValidationError("Cannot perform backward transform with invalid georeferencing")
        map_geom = self.geodetic_to_map(geodetic)
        return self.map_to_pixel(map_geom, shape=shape)

    def evaluate_roundtrip_error(self, original_pixel: PixelGeometry, geodetic: GeodeticGeometry) -> float:
        back_map = self.geodetic_to_map(geodetic)
        back_px = self.map_to_pixel(back_map, shape=original_pixel.shape)
        d_col = original_pixel.col - back_px.col
        d_row = original_pixel.row - back_px.row
        return float(math.hypot(d_col, d_row))

    def pixel_to_geo(self, col: float, row: float) -> Tuple[float, float]:
        if not self.is_valid:
            raise GeoreferenceValidationError("Cannot geolocate coordinates without valid georeference")
        px = PixelGeometry(row=float(row), col=float(col), shape=(0, 0))
        geo, status = self.forward(px, validate_roundtrip=False)
        if status != GeometryStatus.GEOCODED:
            raise GeoreferenceValidationError("Transformation failed to achieve GEOCODED status")
        return geo.lon, geo.lat

    def geo_to_pixel(self, lon: float, lat: float) -> Tuple[float, float]:
        if not self.is_valid:
            raise GeoreferenceValidationError("Cannot invert geolocated coordinates without valid georeference")
        geo = GeodeticGeometry(lon=float(lon), lat=float(lat), crs="EPSG:4326")
        px = self.backward(geo)
        return px.col, px.row
