import time
from typing import Any, Dict, List, Optional, Tuple
import requests


class GfwClient:
    def __init__(
        self,
        endpoint: str = "https://gateway.api.globalfishingwatch.org/v3",
        api_token: Optional[str] = None,
        mock_mode: bool = False,
    ) -> None:
        self.endpoint = endpoint
        self.api_token = api_token
        self.mock_mode = mock_mode

    def query_vessels_report(
        self,
        bbox: Tuple[float, float, float, float],
        date_range: Tuple[str, str],
        confidence_threshold: float = 0.5,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ) -> List[Dict[str, Any]]:
        return self.get_vessel_presence(
            bbox=bbox,
            date_range=date_range,
            confidence_threshold=confidence_threshold,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
        )

    def get_vessel_presence(
        self,
        bbox: Tuple[float, float, float, float],
        date_range: Tuple[str, str],
        confidence_threshold: float = 0.5,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ) -> List[Dict[str, Any]]:
        if self.mock_mode:
            return self._generate_mock_results(bbox, date_range)

        query_url = f"{self.endpoint.rstrip('/')}/4wings/report"
        params = {
            "spatial-resolution": "HIGH",
            "temporal-resolution": "DAILY",
            "start-date": date_range[0],
            "end-date": date_range[1],
            "min-lon": bbox[0],
            "min-lat": bbox[1],
            "max-lon": bbox[2],
            "max-lat": bbox[3],
        }
        headers = {"Accept": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(
                    query_url, params=params, headers=headers, timeout=10.0
                )
                response.raise_for_status()
                data = response.json()
                entries = data.get("entries", [])
                return [e for e in entries if e.get("confidence", 1.0) >= confidence_threshold]
            except Exception:
                if attempt == max_retries:
                    return []
                time.sleep(backoff_factor * (2 ** (attempt - 1)))
        return []

    def _generate_mock_results(
        self,
        bbox: Tuple[float, float, float, float],
        date_range: Tuple[str, str],
    ) -> List[Dict[str, Any]]:
        min_lon, min_lat, max_lon, max_lat = bbox
        mid_lon = (min_lon + max_lon) / 2.0
        mid_lat = (min_lat + max_lat) / 2.0

        return [
            {
                "mmsi": "412000001",
                "vessel_name": "FISHING_VESSEL_ALPHA",
                "flag": "VNM",
                "lat": mid_lat,
                "lon": mid_lon,
                "timestamp": f"{date_range[0]}T12:00:00Z",
                "confidence": 0.92,
                "apparent_fishing_hours": 4.5,
            },
            {
                "mmsi": "412000002",
                "vessel_name": "CARGO_BRAVO",
                "flag": "PAN",
                "lat": mid_lat + 0.05,
                "lon": mid_lon + 0.05,
                "timestamp": f"{date_range[0]}T12:30:00Z",
                "confidence": 0.88,
                "apparent_fishing_hours": 0.0,
            },
        ]
