from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any
from shapely.geometry import box

from src.domain.enums import Capability, SensorType
from src.domain.observation import Observation
from src.domain.provenance import Provenance
from src.ingestion.providers.base import BaseSatelliteProvider


class MockSatelliteProvider(BaseSatelliteProvider):
    name: str = "mock"
    capability: Capability = Capability.MOCK

    def __init__(self, app_env: str | None = None) -> None:
        env = app_env or os.getenv("APP_ENV", "development")
        if env not in ("test", "simulation"):
            raise RuntimeError("MockSatelliteProvider forbidden in production/development environment")
        self.env = env
        self.name = "mock"
        self.capability = Capability.MOCK
        self._last_success: str | None = None

    def search(
        self,
        bbox: tuple[float, float, float, float],
        date_range: tuple[str, str],
        limit: int = 10,
    ) -> list[Observation]:
        if limit <= 0:
            return []
        now = datetime.now(timezone.utc)
        self._last_success = now.isoformat()
        geom = box(bbox[0], bbox[1], bbox[2], bbox[3])
        count = min(limit, 2)
        results: list[Observation] = []
        for i in range(count):
            obs_id = f"mock-obs-{i + 1}"
            prov = Provenance(
                provenance_id=f"prov-{obs_id}",
                source_asset_id=f"asset-{obs_id}",
                acquisition_metadata_ref=f"meta://{obs_id}",
                pipeline_version="2.0.0",
                git_commit="HEAD",
                config_hash="mock-config-hash",
                model_id=None,
                model_version=None,
                input_checksum="mock-input-checksum",
                output_checksum="mock-output-checksum",
                processing_run_id=f"run-{obs_id}",
                processing_time_utc=now,
            )
            obs = Observation(
                observation_id=obs_id,
                source="mock",
                platform="MOCK-SAT-1",
                sensor=SensorType.S1_SAR,
                product="GRD",
                processing_level="LEVEL1",
                acquisition_time_utc=now,
                crs="EPSG:4326",
                ground_sample_distance_m=10.0,
                bounds_wgs84=geom,
                asset_uri=f"file:///tmp/{obs_id}.tif",
                metadata_uri=f"file:///tmp/{obs_id}.xml",
                quality_flags=("SYNTHETIC", "MOCK_DATA"),
                provenance=prov,
            )
            results.append(obs)
        return results

    def fetch_asset(self, observation: Observation, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"SYNTHETIC_MOCK_ASSET_DATA")
        return output_path

    def health_check(self) -> dict[str, Any]:
        now_str = datetime.now(timezone.utc).isoformat()
        return {
            "configured": True,
            "reachable": True,
            "authenticated": True,
            "latency_ms": 0.0,
            "last_success": self._last_success or now_str,
            "data_freshness": "SIMULATED",
        }
