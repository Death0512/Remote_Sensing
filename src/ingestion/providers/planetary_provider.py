from datetime import datetime, timezone
import os
from pathlib import Path
import time
from typing import Any
import requests
from shapely.geometry import box, shape

import planetary_computer
from pystac_client import Client

from src.domain.enums import Capability, SensorType
from src.domain.observation import Observation
from src.domain.provenance import Provenance
from src.ingestion.providers.base import (
    BaseSatelliteProvider,
    ProviderAuthError,
    ProviderNotAvailableError,
    ProviderUnreachableError,
)


class PlanetaryComputerProvider(BaseSatelliteProvider):
    name: str = "planetary_computer"
    capability: Capability = Capability.IMPLEMENTED

    def __init__(
        self,
        endpoint: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
        api_key: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key or os.getenv("PC_SDK_SUBSCRIPTION_KEY") or os.getenv("PLANETARY_COMPUTER_API_KEY")
        self.timeout = timeout
        self.name = "planetary_computer"
        self.capability = Capability.IMPLEMENTED
        self._last_success: str | None = None

    def search(
        self,
        bbox: tuple[float, float, float, float],
        date_range: tuple[str, str],
        limit: int = 10,
        collections: list[str] | None = None,
    ) -> list[Observation]:
        if limit <= 0:
            return []

        search_collections = collections or ["sentinel-1-grd"]
        datetime_str = f"{date_range[0]}/{date_range[1]}"
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Ocp-Apim-Subscription-Key"] = self.api_key

        try:
            modifier = planetary_computer.sign_inplace if planetary_computer is not None else None
            client = Client.open(self.endpoint, headers=headers if headers else None, modifier=modifier)
            search_request = client.search(
                collections=search_collections,
                bbox=list(bbox),
                datetime=datetime_str,
                max_items=limit,
            )
            items = list(search_request.items())
        except Exception as exc:
            err_text = str(exc).lower()
            if "401" in err_text or "403" in err_text or "unauthorized" in err_text or "forbidden" in err_text:
                raise ProviderAuthError(f"Planetary Computer authentication failed: {exc}") from exc
            raise ProviderUnreachableError(f"Planetary Computer search unreachable: {exc}") from exc

        now = datetime.now(timezone.utc)
        self._last_success = now.isoformat()
        results: list[Observation] = []
        for item in items:
            obs_id = item.id
            coll_id = (item.collection_id or "").lower()
            if "sentinel-2" in coll_id:
                sensor = SensorType.S2_MSI
            else:
                sensor = SensorType.S1_SAR

            props = item.properties or {}
            platform = str(props.get("platform", "Sentinel-1"))
            product = str(props.get("sar:product_type") or props.get("processing:software") or "GRD")
            processing_level = str(props.get("s1:processing_level", "LEVEL1"))
            acq_dt = item.datetime or now
            crs = str(props.get("proj:epsg", "EPSG:4326"))
            gsd = float(props.get("sar:pixel_spacing_range") or props.get("gsd") or 10.0)

            if item.geometry:
                geom = shape(item.geometry)
            else:
                geom = box(bbox[0], bbox[1], bbox[2], bbox[3])

            asset_uri = ""
            for key in ("vv", "vh", "visual", "data"):
                if key in item.assets:
                    asset_uri = item.assets[key].href
                    break
            if not asset_uri and item.assets:
                asset_uri = next(iter(item.assets.values())).href

            meta_uri = item.get_self_href() or f"{self.endpoint}/collections/{item.collection_id}/items/{item.id}"

            prov = Provenance(
                provenance_id=f"prov-{obs_id}",
                source_asset_id=obs_id,
                acquisition_metadata_ref=meta_uri,
                pipeline_version="2.0.0",
                git_commit="HEAD",
                config_hash="default",
                model_id=None,
                model_version=None,
                input_checksum=obs_id,
                output_checksum=obs_id,
                processing_run_id=f"run-{obs_id}",
                processing_time_utc=now,
            )

            obs = Observation(
                observation_id=obs_id,
                source="planetary_computer",
                platform=platform,
                sensor=sensor,
                product=product,
                processing_level=processing_level,
                acquisition_time_utc=acq_dt,
                crs=crs,
                ground_sample_distance_m=gsd,
                bounds_wgs84=geom,
                asset_uri=asset_uri,
                metadata_uri=meta_uri,
                quality_flags=(),
                provenance=prov,
            )
            results.append(obs)

        return results

    def fetch_asset(self, observation: Observation, output_path: Path) -> Path:
        if not observation.asset_uri:
            raise ProviderNotAvailableError("Observation asset_uri is empty")

        output_path = Path(output_path)
        if observation.asset_uri.startswith("file://"):
            local_file = Path(observation.asset_uri.replace("file://", ""))
            if not local_file.exists():
                raise ProviderNotAvailableError(f"Local asset not found: {local_file}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(local_file.read_bytes())
            return output_path

        headers: dict[str, str] = {}
        if self.api_key:
            headers["Ocp-Apim-Subscription-Key"] = self.api_key

        try:
            with requests.get(observation.asset_uri, headers=headers, stream=True, timeout=30.0) as resp:
                if resp.status_code in (401, 403):
                    raise ProviderAuthError(f"Planetary Computer fetch asset unauthorized: {resp.status_code}")
                resp.raise_for_status()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
        except ProviderAuthError:
            raise
        except requests.exceptions.RequestException as exc:
            raise ProviderUnreachableError(f"Failed to fetch asset from {observation.asset_uri}: {exc}") from exc

        return output_path

    def health_check(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Ocp-Apim-Subscription-Key"] = self.api_key

        reachable = False
        authenticated = False
        latency_ms: float | None = None
        freshness: str | None = None

        try:
            resp = requests.get(self.endpoint, headers=headers, timeout=self.timeout)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            if resp.status_code == 200:
                reachable = True
                authenticated = True
                self._last_success = datetime.now(timezone.utc).isoformat()
                freshness = "LIVE"
            elif resp.status_code in (401, 403):
                reachable = True
                authenticated = False
            else:
                reachable = False
                authenticated = False
        except Exception:
            latency_ms = None
            reachable = False
            authenticated = False

        return {
            "configured": True,
            "reachable": reachable,
            "authenticated": authenticated,
            "latency_ms": latency_ms,
            "last_success": self._last_success,
            "data_freshness": freshness,
        }
