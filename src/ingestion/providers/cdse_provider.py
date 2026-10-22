from datetime import datetime, timezone
import os
from pathlib import Path
import time
from typing import Any
import requests
from shapely.geometry import box, shape

from src.domain.enums import Capability, SensorType
from src.domain.observation import Observation
from src.domain.provenance import Provenance
from src.ingestion.providers.base import (
    BaseSatelliteProvider,
    ProviderAuthError,
    ProviderNotAvailableError,
    ProviderUnreachableError,
)


class CdseProvider(BaseSatelliteProvider):
    name: str = "cdse"
    capability: Capability = Capability.IMPLEMENTED

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        username: str | None = None,
        password: str | None = None,
        token_url: str = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        odata_url: str = "https://catalogue.dataspace.copernicus.eu/odata/v1",
        stac_url: str = "https://catalogue.dataspace.copernicus.eu/stac",
        timeout: float = 10.0,
    ) -> None:
        self.client_id = client_id or os.getenv("CDSE_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("CDSE_CLIENT_SECRET")
        self.username = username or os.getenv("CDSE_USERNAME")
        self.password = password or os.getenv("CDSE_PASSWORD")
        self.token_url = token_url
        self.odata_url = odata_url.rstrip("/")
        self.stac_url = stac_url.rstrip("/")
        self.timeout = timeout
        self.name = "cdse"
        self.capability = Capability.IMPLEMENTED
        self._access_token: str | None = None
        self._token_expires_at: float = 0.0
        self._last_success: str | None = None

    def _has_credentials(self) -> bool:
        return bool(
            (self.username and self.password)
            or (self.client_id and self.client_secret)
        )

    def _get_token(self) -> str:
        if not self._has_credentials():
            raise ProviderAuthError("CDSE credentials not configured: username/password or client_id/client_secret required")

        if self._access_token and time.time() < self._token_expires_at:
            return self._access_token

        payload: dict[str, str] = {}
        if self.username and self.password:
            payload["grant_type"] = "password"
            payload["client_id"] = self.client_id or "cdse-public"
            payload["username"] = self.username
            payload["password"] = self.password
        elif self.client_secret:
            payload["grant_type"] = "client_credentials"
            payload["client_id"] = self.client_id or "cdse-public"
            payload["client_secret"] = self.client_secret

        try:
            resp = requests.post(self.token_url, data=payload, timeout=self.timeout)
            if resp.status_code in (401, 403):
                raise ProviderAuthError(f"CDSE authentication failed: HTTP {resp.status_code} - {resp.text}")
            resp.raise_for_status()
            data = resp.json()
            token = data.get("access_token")
            if not token:
                raise ProviderAuthError("CDSE token response missing access_token")
            expires_in = float(data.get("expires_in", 300))
            self._access_token = token
            self._token_expires_at = time.time() + expires_in - 30.0
            return token
        except ProviderAuthError:
            raise
        except requests.exceptions.RequestException as exc:
            raise ProviderUnreachableError(f"CDSE token endpoint unreachable: {exc}") from exc

    def search(
        self,
        bbox: tuple[float, float, float, float],
        date_range: tuple[str, str],
        limit: int = 10,
    ) -> list[Observation]:
        if not self._has_credentials():
            raise ProviderAuthError("CDSE credentials not configured")

        if limit <= 0:
            return []

        token = self._get_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        search_url = f"{self.stac_url}/search"
        payload = {
            "collections": ["SENTINEL-1"],
            "bbox": list(bbox),
            "datetime": f"{date_range[0]}/{date_range[1]}",
            "limit": limit,
        }

        try:
            resp = requests.post(search_url, json=payload, headers=headers, timeout=self.timeout)
            if resp.status_code in (401, 403):
                raise ProviderAuthError(f"CDSE search unauthorized: {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
        except ProviderAuthError:
            raise
        except requests.exceptions.RequestException as exc:
            raise ProviderUnreachableError(f"CDSE search unreachable: {exc}") from exc

        features = data.get("features", [])
        now = datetime.now(timezone.utc)
        self._last_success = now.isoformat()
        results: list[Observation] = []

        for feat in features:
            feat_id = feat.get("id", "cdse-product")
            props = feat.get("properties", {})
            dt_str = props.get("datetime")
            acq_time = datetime.fromisoformat(dt_str.replace("Z", "+00:00")) if dt_str else now
            geom_dict = feat.get("geometry")
            geom = shape(geom_dict) if geom_dict else box(bbox[0], bbox[1], bbox[2], bbox[3])
            assets = feat.get("assets", {})
            asset_uri = ""
            if "data" in assets:
                asset_uri = assets["data"].get("href", "")
            elif assets:
                asset_uri = next(iter(assets.values())).get("href", "")
            if not asset_uri:
                asset_uri = f"{self.odata_url}/Products({feat_id})/$value"

            meta_uri = f"{self.odata_url}/Products({feat_id})"

            prov = Provenance(
                provenance_id=f"prov-{feat_id}",
                source_asset_id=feat_id,
                acquisition_metadata_ref=meta_uri,
                pipeline_version="2.0.0",
                git_commit="HEAD",
                config_hash="default",
                model_id=None,
                model_version=None,
                input_checksum=feat_id,
                output_checksum=feat_id,
                processing_run_id=f"run-{feat_id}",
                processing_time_utc=now,
            )

            obs = Observation(
                observation_id=feat_id,
                source="cdse",
                platform=str(props.get("platform", "Sentinel-1")),
                sensor=SensorType.S1_SAR,
                product=str(props.get("sar:product_type", "GRD")),
                processing_level="LEVEL1",
                acquisition_time_utc=acq_time,
                crs=str(props.get("proj:epsg", "EPSG:4326")),
                ground_sample_distance_m=float(props.get("sar:pixel_spacing_range", 10.0)),
                bounds_wgs84=geom,
                asset_uri=asset_uri,
                metadata_uri=meta_uri,
                quality_flags=(),
                provenance=prov,
            )
            results.append(obs)

        return results

    def fetch_asset(self, observation: Observation, output_path: Path) -> Path:
        if not self._has_credentials():
            raise ProviderAuthError("CDSE credentials not configured")

        if not observation.asset_uri:
            raise ProviderNotAvailableError("Observation asset_uri is empty")

        token = self._get_token()
        headers = {"Authorization": f"Bearer {token}"}
        output_path = Path(output_path)

        try:
            with requests.get(observation.asset_uri, headers=headers, stream=True, timeout=60.0) as resp:
                if resp.status_code in (401, 403):
                    raise ProviderAuthError(f"CDSE fetch unauthorized: {resp.status_code}")
                resp.raise_for_status()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
        except ProviderAuthError:
            raise
        except requests.exceptions.RequestException as exc:
            raise ProviderUnreachableError(f"CDSE fetch unreachable: {exc}") from exc

        return output_path

    def health_check(self) -> dict[str, Any]:
        configured = self._has_credentials()
        t0 = time.perf_counter()
        reachable = False
        latency_ms: float | None = None
        try:
            resp = requests.get(self.stac_url, timeout=self.timeout)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            if resp.status_code == 200:
                reachable = True
        except Exception:
            reachable = False
            latency_ms = None

        authenticated = False
        if configured:
            try:
                self._get_token()
                authenticated = True
                self._last_success = datetime.now(timezone.utc).isoformat()
            except Exception:
                authenticated = False

        return {
            "configured": configured,
            "reachable": reachable,
            "authenticated": authenticated,
            "latency_ms": latency_ms,
            "last_success": self._last_success,
            "data_freshness": "LIVE" if (reachable and authenticated) else None,
        }
