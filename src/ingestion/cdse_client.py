from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import requests


class TokenManager:
    def __init__(
        self,
        token_url: str = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_id: str = "cdse-public",
        client_secret: Optional[str] = None,
    ) -> None:
        self.token_url = token_url
        self.username = username
        self.password = password
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token: Optional[str] = None
        self.expires_at: float = 0.0

    def get_token(self) -> str:
        if self.access_token and time.time() < self.expires_at:
            return self.access_token

        if not self.username and not self.client_secret:
            return "mock_token"

        payload: Dict[str, str] = {"client_id": self.client_id}
        if self.password and self.username:
            payload["grant_type"] = "password"
            payload["username"] = self.username
            payload["password"] = self.password
        elif self.client_secret:
            payload["grant_type"] = "client_credentials"
            payload["client_secret"] = self.client_secret

        try:
            resp = requests.post(self.token_url, data=payload, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            self.access_token = data.get("access_token")
            expires_in = data.get("expires_in", 300)
            self.expires_at = time.time() + float(expires_in) - 30.0
            return self.access_token or "mock_token"
        except Exception:
            return "mock_token"


class CopernicusClient:
    def __init__(
        self,
        endpoint: str = "https://catalogue.dataspace.copernicus.eu/stac",
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        mock_mode: bool = False,
    ) -> None:
        self.endpoint = endpoint
        self.client_id = client_id
        self.client_secret = client_secret
        self.mock_mode = mock_mode
        self.token_manager = TokenManager(
            username=username,
            password=password,
            client_id=client_id or "cdse-public",
            client_secret=client_secret,
        )

    def search_sentinel1(
        self,
        bbox: Tuple[float, float, float, float],
        date_range: Tuple[str, str],
        product_type: str = "GRD",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        if self.mock_mode:
            return self._generate_mock_results(bbox, date_range, product_type, limit)

        search_url = f"{self.endpoint.rstrip('/')}/search"
        payload = {
            "collections": ["SENTINEL-1"],
            "bbox": list(bbox),
            "datetime": f"{date_range[0]}/{date_range[1]}",
            "limit": limit,
            "query": {"productType": {"eq": product_type}},
        }
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(
                search_url, json=payload, headers=headers, timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            return data.get("features", [])
        except Exception:
            return []

    def search_sentinel2_l1c(
        self,
        bbox: Tuple[float, float, float, float],
        date_range: Tuple[str, str],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        if self.mock_mode:
            return self._generate_mock_s2_results(bbox, date_range, limit)

        search_url = f"{self.endpoint.rstrip('/')}/search"
        payload = {
            "collections": ["SENTINEL-2"],
            "bbox": list(bbox),
            "datetime": f"{date_range[0]}/{date_range[1]}",
            "limit": limit,
            "query": {"productType": {"eq": "S2MSI1C"}},
        }
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(
                search_url, json=payload, headers=headers, timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            return data.get("features", [])
        except Exception:
            return []

    def download_product(
        self,
        product_id: str,
        output_path: Union[str, Path],
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            return out

        if self.mock_mode:
            out.write_bytes(b"MOCK_S2_PRODUCT_DATA")
            return out

        download_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
        token = self.token_manager.get_token()
        headers = {"Authorization": f"Bearer {token}"}
        part_path = out.with_suffix(out.suffix + ".part")

        for attempt in range(1, max_retries + 1):
            try:
                if part_path.exists():
                    part_path.unlink()
                with requests.get(download_url, headers=headers, stream=True, timeout=30.0) as resp:
                    resp.raise_for_status()
                    with open(part_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                part_path.replace(out)
                return out
            except Exception as exc:
                if part_path.exists():
                    part_path.unlink()
                if attempt == max_retries:
                    raise exc
                time.sleep(backoff_factor * (2 ** (attempt - 1)))
        return out

    def _generate_mock_results(
        self,
        bbox: Tuple[float, float, float, float],
        date_range: Tuple[str, str],
        product_type: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        count = min(limit, 3)
        min_lon, min_lat, max_lon, max_lat = bbox

        for i in range(count):
            scene_id = f"S1A_IW_{product_type}_1SDV_{date_range[0][:10].replace('-', '')}_{i:03d}"
            results.append(
                {
                    "id": scene_id,
                    "type": "Feature",
                    "collection": "SENTINEL-1",
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
                        "productType": product_type,
                        "platform": "Sentinel-1A",
                    },
                    "assets": {
                        "quicklook": {
                            "href": f"https://mock.copernicus.eu/quicklook/{scene_id}.png",
                            "type": "image/png",
                        },
                        "download": {
                            "href": f"https://mock.copernicus.eu/odata/v1/Products({scene_id})/$value",
                            "type": "application/octet-stream",
                        },
                    },
                }
            )
        return results

    def _generate_mock_s2_results(
        self,
        bbox: Tuple[float, float, float, float],
        date_range: Tuple[str, str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        count = min(limit, 3)
        min_lon, min_lat, max_lon, max_lat = bbox

        for i in range(count):
            scene_id = f"S2A_MSIL1C_{date_range[0][:10].replace('-', '')}_{i:03d}"
            results.append(
                {
                    "id": scene_id,
                    "type": "Feature",
                    "collection": "SENTINEL-2",
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
                        "productType": "S2MSI1C",
                        "platform": "Sentinel-2A",
                    },
                    "assets": {
                        "download": {
                            "href": f"https://mock.copernicus.eu/odata/v1/Products({scene_id})/$value",
                            "type": "application/octet-stream",
                        },
                    },
                }
            )
        return results
