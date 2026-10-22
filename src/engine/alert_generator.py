import dataclasses
from pathlib import Path
import struct
from typing import Any, Literal, Optional, Tuple, Union
import zlib
import numpy as np
import shapely.geometry

from src.core.contracts import Alert, Detection

try:
    import rasterio
except ImportError:
    rasterio = None


class AlertGenerator:
    def __init__(
        self,
        output_dir: Union[str, Path] = "data/outputs/chips",
        chip_size: int = 256,
        padding_mode: str = "reflect",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chip_size = chip_size
        self.padding_mode = padding_mode

    def _write_png(self, file_path: Path, img_array: np.ndarray) -> None:
        if img_array.dtype != np.uint8:
            min_v = float(img_array.min())
            max_v = float(img_array.max())
            if max_v > min_v:
                arr_u8 = ((img_array - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
            else:
                arr_u8 = np.clip(img_array, 0, 255).astype(np.uint8)
        else:
            arr_u8 = img_array

        if arr_u8.ndim == 3 and arr_u8.shape[0] in (1, 3, 4) and arr_u8.shape[1] > 4:
            arr_u8 = np.transpose(arr_u8, (1, 2, 0))

        if arr_u8.ndim == 2:
            height, width = arr_u8.shape
            color_type = 0
            raw = bytearray()
            for y in range(height):
                raw.append(0)
                raw.extend(arr_u8[y, :].tobytes())
        elif arr_u8.ndim == 3:
            if arr_u8.shape[2] == 1:
                flat = arr_u8[:, :, 0]
                height, width = flat.shape
                color_type = 0
                raw = bytearray()
                for y in range(height):
                    raw.append(0)
                    raw.extend(flat[y, :].tobytes())
            else:
                rgb = arr_u8[:, :, :3]
                height, width, _ = rgb.shape
                color_type = 2
                raw = bytearray()
                for y in range(height):
                    raw.append(0)
                    raw.extend(rgb[y, :, :].tobytes())
        else:
            raise ValueError(f"Unsupported array shape: {arr_u8.shape}")

        compressed = zlib.compress(bytes(raw))

        def chunk(tag: bytes, payload: bytes) -> bytes:
            crc = zlib.crc32(tag + payload)
            return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

        png_bytes = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
            + chunk(b"IDAT", compressed)
            + chunk(b"IEND", b"")
        )
        with open(file_path, "wb") as f:
            f.write(png_bytes)

    def extract_chip(
        self,
        source_image: Union[str, Path, np.ndarray, Any],
        geom_px: Tuple[float, float, float, float],
    ) -> np.ndarray:
        if isinstance(source_image, (str, Path)):
            if rasterio is None:
                raise ImportError("rasterio required to read image from path")
            with rasterio.open(str(source_image)) as src:
                data = src.read()
        elif hasattr(source_image, "read"):
            data = source_image.read()
        elif isinstance(source_image, np.ndarray):
            data = source_image
        else:
            raise ValueError(f"Unsupported source image type: {type(source_image)}")

        is_channels_first = False
        if data.ndim == 3:
            if data.shape[0] in (1, 3, 4) and data.shape[1] > 4:
                is_channels_first = True
                channels, h, w = data.shape
            else:
                h, w, channels = data.shape
        elif data.ndim == 2:
            h, w = data.shape
        else:
            raise ValueError(f"Unsupported data ndim: {data.ndim}")

        c_center = int(round((geom_px[0] + geom_px[2]) / 2.0))
        r_center = int(round((geom_px[1] + geom_px[3]) / 2.0))
        half_size = self.chip_size // 2

        r_start = r_center - half_size
        r_end = r_start + self.chip_size
        c_start = c_center - half_size
        c_end = c_start + self.chip_size

        pad_top = max(0, -r_start)
        pad_bottom = max(0, r_end - h)
        pad_left = max(0, -c_start)
        pad_right = max(0, c_end - w)

        v_r_start = max(0, r_start)
        v_r_end = min(h, r_end)
        v_c_start = max(0, c_start)
        v_c_end = min(w, c_end)

        if v_r_end <= v_r_start or v_c_end <= v_c_start:
            if data.ndim == 2:
                return np.zeros((self.chip_size, self.chip_size), dtype=data.dtype)
            if is_channels_first:
                return np.zeros((channels, self.chip_size, self.chip_size), dtype=data.dtype)
            return np.zeros((self.chip_size, self.chip_size, channels), dtype=data.dtype)

        if data.ndim == 2:
            valid_patch = data[v_r_start:v_r_end, v_c_start:v_c_end]
        elif is_channels_first:
            valid_patch = data[:, v_r_start:v_r_end, v_c_start:v_c_end]
        else:
            valid_patch = data[v_r_start:v_r_end, v_c_start:v_c_end, :]

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            can_reflect = False
            if self.padding_mode == "reflect":
                if data.ndim == 2:
                    can_reflect = (
                        valid_patch.shape[0] > max(pad_top, pad_bottom)
                        and valid_patch.shape[1] > max(pad_left, pad_right)
                    )
                elif is_channels_first:
                    can_reflect = (
                        valid_patch.shape[1] > max(pad_top, pad_bottom)
                        and valid_patch.shape[2] > max(pad_left, pad_right)
                    )
                else:
                    can_reflect = (
                        valid_patch.shape[0] > max(pad_top, pad_bottom)
                        and valid_patch.shape[1] > max(pad_left, pad_right)
                    )

            mode = "reflect" if can_reflect else "constant"

            if data.ndim == 2:
                pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
            elif is_channels_first:
                pad_width = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
            else:
                pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

            if mode == "constant":
                chip = np.pad(valid_patch, pad_width, mode="constant", constant_values=0)
            else:
                chip = np.pad(valid_patch, pad_width, mode="reflect")
        else:
            chip = valid_patch

        return chip

    def generate_alert(
        self,
        detection: Detection,
        zone: Literal[1, 2, 3],
        score_s: float,
        source_image: Union[str, Path, np.ndarray, Any],
        is_dark: Optional[bool] = None,
        alert_id: Optional[str] = None,
    ) -> Alert:
        if zone == 1:
            level: Literal["CRITICAL", "HIGH", "STRATEGIC"] = "CRITICAL"
        elif zone == 2:
            level = "HIGH"
        elif zone == 3:
            level = "STRATEGIC"
        else:
            raise ValueError(f"Invalid zone: {zone}, must be 1, 2, or 3")

        assigned_id = alert_id or f"ALT-{detection.det_id}"
        chip = self.extract_chip(source_image, detection.geom_px)
        chip_filename = f"chip_{assigned_id}.png"
        chip_path = self.output_dir / chip_filename
        self._write_png(chip_path, chip)

        geojson_feature = {
            "type": "Feature",
            "geometry": shapely.geometry.mapping(detection.geom_wgs84),
            "properties": {
                "alert_id": assigned_id,
                "det_id": detection.det_id,
                "target_class": detection.target_class,
                "confidence": float(detection.confidence),
                "sensor": detection.sensor,
                "score_s": float(score_s),
                "zone": zone,
                "level": level,
                "acquisition_time": detection.acquisition_time.isoformat(),
                "is_dark": is_dark,
            },
        }

        return Alert(
            alert_id=assigned_id,
            zone=zone,
            level=level,
            score_s=float(score_s),
            is_dark=is_dark,
            geojson_feature=geojson_feature,
            chip_path=str(chip_path.resolve()),
            audit_status="PENDING",
        )

    def update_audit_status(
        self,
        alert: Alert,
        status: Literal["CONFIRMED", "DISMISSED"],
    ) -> Alert:
        if status not in ("CONFIRMED", "DISMISSED"):
            raise ValueError(f"Invalid audit status: {status}")
        return dataclasses.replace(alert, audit_status=status)
