import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import requests

try:
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds
except ImportError:
    rasterio = None
    from_bounds = None
    transform_bounds = None

try:
    import planetary_computer
except ImportError:
    planetary_computer = None


class PlanetaryComputerClient:
    def __init__(
        self,
        endpoint: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
        api_key: Optional[str] = None,
        mock_mode: bool = False,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.mock_mode = mock_mode

    def search_scenes(
        self,
        bbox: Tuple[float, float, float, float],
        date_range: Tuple[str, str],
        collections: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        if collections is None:
            collections = ["sentinel-1-rtc"]

        if self.mock_mode:
            return self._generate_mock_results(bbox, date_range, collections, limit)

        search_url = f"{self.endpoint.rstrip('/')}/search"
        payload = {
            "collections": collections,
            "bbox": list(bbox),
            "datetime": f"{date_range[0]}/{date_range[1]}",
            "limit": limit,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Ocp-Apim-Subscription-Key"] = self.api_key

        try:
            response = requests.post(
                search_url, json=payload, headers=headers, timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            return data.get("features", [])
        except Exception:
            return []

    def fetch_scene_window(
        self,
        item_id: Union[str, Dict[str, Any]],
        bbox: Optional[Tuple[float, float, float, float]] = None,
        aoi_bounds: Optional[Tuple[float, float, float, float]] = None,
        asset_key: str = "vv",
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ) -> Optional[np.ndarray]:
        if self.mock_mode:
            return np.ones((256, 256), dtype=np.float32) * 0.15

        if rasterio is None:
            return None

        item: Optional[Dict[str, Any]] = None
        if isinstance(item_id, dict):
            item = item_id
        else:
            search_url = f"{self.endpoint.rstrip('/')}/collections/sentinel-1-rtc/items/{item_id}"
            headers = {}
            if self.api_key:
                headers["Ocp-Apim-Subscription-Key"] = self.api_key
            try:
                resp = requests.get(search_url, headers=headers, timeout=10.0)
                if resp.status_code == 200:
                    item = resp.json()
            except Exception:
                pass

        if not item:
            return None

        if planetary_computer is not None:
            try:
                signed_item = planetary_computer.sign(item)
            except Exception:
                signed_item = item
        else:
            signed_item = item

        assets = signed_item.get("assets", {})
        if asset_key not in assets:
            return None

        href = assets[asset_key].get("href")
        if not href:
            return None

        for attempt in range(1, max_retries + 1):
            try:
                with rasterio.Env(
                    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                    GDAL_HTTP_TIMEOUT="15",
                    GDAL_HTTP_MAX_RETRY="2",
                    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
                ):
                    with rasterio.open(href) as src:
                        if aoi_bounds and from_bounds is not None:
                            if src.crs and str(src.crs) != "EPSG:4326" and transform_bounds is not None:
                                try:
                                    proj_bounds = transform_bounds("EPSG:4326", src.crs, *aoi_bounds)
                                    window = from_bounds(*proj_bounds, transform=src.transform)
                                except Exception:
                                    window = from_bounds(*aoi_bounds, transform=src.transform)
                            else:
                                window = from_bounds(*aoi_bounds, transform=src.transform)

                            win_h = max(1, int(round(window.height)))
                            win_w = max(1, int(round(window.width)))
                            if win_h > 1024 or win_w > 1024:
                                scale = min(1024 / win_h, 1024 / win_w)
                                out_shape = (
                                    max(1, int(round(win_h * scale))),
                                    max(1, int(round(win_w * scale))),
                                )
                                arr = src.read(1, window=window, out_shape=out_shape)
                            else:
                                arr = src.read(1, window=window)
                        else:
                            arr = src.read(1, out_shape=(512, 512))

                        if arr is not None and arr.size > 0 and arr.shape[-2] > 0 and arr.shape[-1] > 0:
                            return arr.astype(np.float32)
                        return None
            except Exception:
                if attempt == max_retries:
                    return None
                time.sleep(backoff_factor * (2 ** (attempt - 1)))

        return None

    def _generate_mock_results(
        self,
        bbox: Tuple[float, float, float, float],
        date_range: Tuple[str, str],
        collections: List[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        count = min(limit, 3)
        min_lon, min_lat, max_lon, max_lat = bbox

        for i in range(count):
            scene_id = f"S1A_IW_RTC_MOCK_{date_range[0][:10].replace('-', '')}_{i:03d}"
            results.append(
                {
                    "id": scene_id,
                    "type": "Feature",
                    "collection": collections[0],
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [min_lon, min_lat],
                                [max_lon, min_lat],
                                [max_lon, max_lat],
                                [min_lon, max_lat],
                                [min_lon, min_lat],
                            ]
                        ],
                    },
                    "properties": {
                        "datetime": f"{date_range[0]}T00:00:00Z",
                        "sar:instrument_mode": "IW",
                        "sar:polarizations": ["VV", "VH"],
                    },
                    "assets": {
                        "vv": {
                            "href": f"https://mock.storage/rtc/{scene_id}_vv.tif",
                            "type": "image/tiff; application=geotiff",
                        },
                        "vh": {
                            "href": f"https://mock.storage/rtc/{scene_id}_vh.tif",
                            "type": "image/tiff; application=geotiff",
                        },
                    },
                }
            )
        return results
