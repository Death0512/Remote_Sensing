from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence, Union
from pyproj import CRS
from pyproj.exceptions import CRSError

from src.domain.enums import SensorType
from src.domain.observation import Observation
from src.ingestion.exceptions import (
    DataCorruptedError,
    DataUnavailableError,
    InvalidGeoreferenceError,
    InvalidMetadataError,
)

SOURCE_RULES: dict[str, dict[SensorType, set[str]]] = {
    "cdse": {
        SensorType.S1_SAR: {"GRD", "SLC", "RAW", "OCN"},
        SensorType.S2_MSI: {"L1C", "L2A", "S2MSI1C", "S2MSI2A"},
    },
    "planetary_computer": {
        SensorType.S1_SAR: {"GRD", "RTC", "SENTINEL-1-GRD", "SENTINEL-1-RTC"},
        SensorType.S2_MSI: {"L2A", "SENTINEL-2-L2A"},
    },
    "vhr": {
        SensorType.VHR_OPTICAL: {"PAN", "MS", "PS", "RGB", "BUNDLE", "VHR_BASIC", "VHR_ORTHO"},
    },
    "mock": {
        SensorType.S1_SAR: {"GRD", "RTC"},
        SensorType.S2_MSI: {"L2A", "L1C"},
        SensorType.VHR_OPTICAL: {"RGB", "PAN"},
    },
}


class MetadataValidator:
    @staticmethod
    def validate_crs(crs_val: Any) -> CRS:
        if not crs_val:
            raise InvalidGeoreferenceError("CRS is required and cannot be empty")
        try:
            return CRS.from_user_input(crs_val)
        except Exception as exc:
            raise InvalidGeoreferenceError(
                f"Declared CRS '{crs_val}' cannot be resolved in PROJ/pyproj database: {exc}"
            ) from exc

    @staticmethod
    def validate_affine_transform(
        transform: Any,
    ) -> tuple[float, float, float, float, float, float]:
        if transform is None:
            raise InvalidGeoreferenceError("Affine geotransform is required and cannot be None")

        if hasattr(transform, "to_gdal"):
            g = transform.to_gdal()
            transform_seq = (g[1], g[2], g[0], g[4], g[5], g[3])
        elif isinstance(transform, (tuple, list)):
            transform_seq = tuple(transform)
        else:
            raise InvalidGeoreferenceError(
                f"Affine geotransform must be a tuple, list, or Affine object, got {type(transform)}"
            )

        if len(transform_seq) != 6:
            raise InvalidGeoreferenceError(
                f"Affine geotransform must have 6 coefficients, got {len(transform_seq)}"
            )

        try:
            a, b, c, d, e, f = [float(v) for v in transform_seq]
        except (ValueError, TypeError) as exc:
            raise InvalidGeoreferenceError(f"Affine coefficients must be numeric: {exc}") from exc

        det = a * e - b * d
        if abs(det) < 1e-12:
            raise InvalidGeoreferenceError(
                f"Affine geotransform is degenerate with determinant {det}"
            )

        is_identity = (
            math.isclose(a, 1.0, abs_tol=1e-7)
            and math.isclose(b, 0.0, abs_tol=1e-7)
            and math.isclose(c, 0.0, abs_tol=1e-7)
            and math.isclose(d, 0.0, abs_tol=1e-7)
            and (math.isclose(e, 1.0, abs_tol=1e-7) or math.isclose(e, -1.0, abs_tol=1e-7))
            and math.isclose(f, 0.0, abs_tol=1e-7)
        )
        if is_identity:
            raise InvalidGeoreferenceError(
                "Affine geotransform is an unreferenced identity matrix"
            )

        return (a, b, c, d, e, f)

    @staticmethod
    def validate_gsd(
        gsd_val: Any,
        transform: tuple[float, float, float, float, float, float],
        crs_obj: CRS,
        tolerance_ratio: float = 0.5,
    ) -> float:
        if gsd_val is None:
            raise InvalidMetadataError("Ground sample distance is required")
        try:
            gsd = float(gsd_val)
        except (ValueError, TypeError) as exc:
            raise InvalidMetadataError(f"Ground sample distance must be a valid float: {exc}") from exc

        if gsd <= 0.0 or math.isnan(gsd) or math.isinf(gsd):
            raise InvalidMetadataError(f"Ground sample distance must be positive and finite, got {gsd}")

        a, b, _, d, e, _ = transform
        sx = math.hypot(a, d)
        sy = math.hypot(b, e)

        if crs_obj.is_geographic:
            affine_scale = ((sx + sy) / 2.0) * 111319.49
        else:
            affine_scale = (sx + sy) / 2.0

        rel_diff = abs(gsd - affine_scale) / max(gsd, affine_scale)
        if rel_diff > tolerance_ratio:
            raise InvalidMetadataError(
                f"Ground sample distance {gsd}m is inconsistent with affine scale {affine_scale:.2f}m (rel_diff={rel_diff:.2f})"
            )

        return gsd

    @staticmethod
    def _parse_utc_datetime(val: Any, field_name: str) -> datetime:
        if val is None:
            raise InvalidMetadataError(f"{field_name} is required")

        if isinstance(val, datetime):
            dt = val
        elif isinstance(val, str):
            clean_str = val.strip().replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(clean_str)
            except Exception as exc:
                raise InvalidMetadataError(
                    f"{field_name} must be a valid ISO 8601 UTC datetime string: {exc}"
                ) from exc
        else:
            raise InvalidMetadataError(
                f"{field_name} must be a datetime or ISO string, got {type(val)}"
            )

        if dt.tzinfo is None:
            raise InvalidMetadataError(f"{field_name} must be a timezone-aware UTC datetime")

        utc_offset = dt.utcoffset()
        if utc_offset is None or utc_offset.total_seconds() != 0:
            raise InvalidMetadataError(f"{field_name} timezone must be UTC (offset 0)")

        return dt

    @staticmethod
    def validate_acquisitions(
        start_val: Any,
        end_val: Any = None,
    ) -> tuple[datetime, datetime | None]:
        start_dt = MetadataValidator._parse_utc_datetime(start_val, "Acquisition start")
        end_dt: datetime | None = None
        if end_val is not None:
            end_dt = MetadataValidator._parse_utc_datetime(end_val, "Acquisition end")
            if end_dt < start_dt:
                raise InvalidMetadataError(
                    f"Acquisition end ({end_dt.isoformat()}) cannot be earlier than start ({start_dt.isoformat()})"
                )

        return start_dt, end_dt

    @staticmethod
    def validate_source_sensor_product(
        source_val: Any,
        sensor_val: Any,
        product_val: Any,
        allow_mock: bool = False,
    ) -> tuple[str, SensorType, str]:
        if not source_val:
            raise InvalidMetadataError("Declared source is required")

        source = str(source_val).strip().lower().replace("-", "_")
        if source in ("sentinel1", "sentinel_1"):
            source = "cdse"
        elif source in ("sentinel2", "sentinel_2"):
            source = "cdse"

        if source not in SOURCE_RULES:
            raise InvalidMetadataError(f"Unsupported declared source: '{source}'")

        if source == "mock":
            current_env = os.getenv("APP_ENV", "development")
            if not allow_mock and current_env not in ("test", "simulation"):
                raise InvalidMetadataError("Mock source forbidden in production/development environment")

        if not sensor_val:
            raise InvalidMetadataError("Sensor is required")

        sensor: SensorType
        if isinstance(sensor_val, SensorType):
            sensor = sensor_val
        elif isinstance(sensor_val, str):
            try:
                sensor = SensorType(sensor_val.strip())
            except ValueError:
                val_upper = sensor_val.strip().upper()
                if "S1" in val_upper or "SAR" in val_upper:
                    sensor = SensorType.S1_SAR
                elif "S2" in val_upper or "MSI" in val_upper:
                    sensor = SensorType.S2_MSI
                elif "VHR" in val_upper or "OPTICAL" in val_upper:
                    sensor = SensorType.VHR_OPTICAL
                else:
                    raise InvalidMetadataError(f"Unknown sensor type: '{sensor_val}'")
        else:
            raise InvalidMetadataError(f"Invalid sensor format: {sensor_val}")

        source_rule = SOURCE_RULES[source]
        if sensor not in source_rule:
            raise InvalidMetadataError(
                f"Sensor '{sensor.value}' is not valid for declared source '{source}'"
            )

        if not product_val:
            raise InvalidMetadataError("Product type is required")

        product = str(product_val).strip().upper()
        allowed_products = source_rule[sensor]
        if product not in allowed_products:
            raise InvalidMetadataError(
                f"Product '{product}' does not match source '{source}' and sensor '{sensor.value}'"
            )

        return source, sensor, product

    @classmethod
    def validate(
        cls,
        metadata: Union[dict[str, Any], Observation],
        affine_transform: Any = None,
        allow_mock: bool = False,
        tolerance_ratio: float = 0.5,
    ) -> dict[str, Any]:
        if isinstance(metadata, Observation):
            crs_input = metadata.crs
            gsd_input = metadata.ground_sample_distance_m
            start_input = metadata.acquisition_time_utc
            end_input = getattr(metadata, "acquisition_end_utc", None)
            source_input = metadata.source
            sensor_input = metadata.sensor
            product_input = metadata.product
            transform_input = affine_transform or getattr(metadata, "affine_transform", None)
        elif isinstance(metadata, dict):
            crs_input = metadata.get("crs") or metadata.get("proj:epsg")
            gsd_input = metadata.get("ground_sample_distance_m") or metadata.get("gsd") or metadata.get("sar:pixel_spacing_range")
            start_input = (
                metadata.get("acquisition_time_utc")
                or metadata.get("acquisition_start")
                or metadata.get("datetime")
                or metadata.get("start_datetime")
            )
            end_input = metadata.get("acquisition_end") or metadata.get("end_datetime")
            source_input = metadata.get("source")
            sensor_input = metadata.get("sensor") or metadata.get("platform")
            product_input = metadata.get("product") or metadata.get("sar:product_type") or metadata.get("product_type")
            transform_input = affine_transform or metadata.get("affine_transform") or metadata.get("transform") or metadata.get("proj:transform")
        else:
            raise InvalidMetadataError(f"Expected dict or Observation, got {type(metadata)}")

        crs_obj = cls.validate_crs(crs_input)
        transform_tuple = cls.validate_affine_transform(transform_input)
        gsd = cls.validate_gsd(gsd_input, transform_tuple, crs_obj, tolerance_ratio=tolerance_ratio)
        start_dt, end_dt = cls.validate_acquisitions(start_input, end_input)
        source, sensor, product = cls.validate_source_sensor_product(
            source_input, sensor_input, product_input, allow_mock=allow_mock
        )

        return {
            "source": source,
            "sensor": sensor,
            "product": product,
            "crs": str(crs_obj),
            "affine_transform": transform_tuple,
            "ground_sample_distance_m": gsd,
            "acquisition_start": start_dt,
            "acquisition_end": end_dt,
            "is_valid": True,
        }

    @classmethod
    def validate_file(
        cls,
        file_path: Union[str, Path],
        expected_sha256: str | None = None,
        affine_transform: Any = None,
        allow_mock: bool = False,
    ) -> dict[str, Any]:
        path = Path(file_path)
        if not path.is_file():
            raise DataUnavailableError(f"Metadata file not found: {path}")

        try:
            content = path.read_bytes()
        except OSError as exc:
            raise DataUnavailableError(f"Cannot read metadata file {path}: {exc}") from exc

        if expected_sha256:
            actual_sha = hashlib.sha256(content).hexdigest().lower()
            if actual_sha != expected_sha256.strip().lower():
                raise DataCorruptedError(
                    f"Metadata file checksum mismatch: expected {expected_sha256}, got {actual_sha}"
                )

        try:
            data = json.loads(content.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DataCorruptedError(f"Metadata file corrupted or invalid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise InvalidMetadataError(f"Metadata JSON must be an object, got {type(data)}")

        return cls.validate(data, affine_transform=affine_transform, allow_mock=allow_mock)
