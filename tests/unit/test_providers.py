from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from src.api.server import app, init_db
from src.domain.enums import Capability, SensorType
from src.domain.observation import Observation
from src.domain.provenance import Provenance
from src.ingestion.providers.base import (
    BaseSatelliteProvider,
    ProviderAuthError,
    ProviderNotAvailableError,
    ProviderUnreachableError,
)
from src.ingestion.providers.cdse_provider import CdseProvider
from src.ingestion.providers.factory import get_provider
from src.ingestion.providers.mock_provider import MockSatelliteProvider
from src.ingestion.providers.planetary_provider import PlanetaryComputerProvider


def test_base_provider_abstract() -> None:
    with pytest.raises(TypeError):
        BaseSatelliteProvider()


def test_mock_provider_rejected_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    with pytest.raises(RuntimeError, match="MockSatelliteProvider forbidden in production/development environment"):
        MockSatelliteProvider()


def test_mock_provider_allowed_in_test(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    provider = MockSatelliteProvider()
    assert provider.name == "mock"
    assert provider.capability == Capability.MOCK

    results = provider.search(
        bbox=(108.0, 15.0, 109.0, 16.0),
        date_range=("2024-01-01", "2024-01-02"),
        limit=2,
    )
    assert len(results) == 2
    obs = results[0]
    assert isinstance(obs, Observation)
    assert obs.sensor == SensorType.S1_SAR
    assert obs.source == "mock"
    assert obs.bounds_wgs84 is not None
    assert isinstance(obs.provenance, Provenance)
    assert obs.provenance.provenance_id == f"prov-{obs.observation_id}"
    assert obs.provenance.git_commit == "HEAD"
    assert obs.provenance.processing_time_utc is not None

    asset_out = tmp_path / "asset.dat"
    fetched = provider.fetch_asset(obs, asset_out)
    assert fetched.exists()
    assert fetched.read_bytes() == b"SYNTHETIC_MOCK_ASSET_DATA"

    health = provider.health_check()
    assert health["configured"] is True
    assert health["reachable"] is True
    assert health["authenticated"] is True
    assert health["latency_ms"] == 0.0
    assert health["data_freshness"] == "SIMULATED"


def test_cdse_fails_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CDSE_CLIENT_ID", raising=False)
    monkeypatch.delenv("CDSE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("CDSE_USERNAME", raising=False)
    monkeypatch.delenv("CDSE_PASSWORD", raising=False)

    provider = CdseProvider(client_id=None, client_secret=None, username=None, password=None)
    with pytest.raises(ProviderAuthError, match="CDSE credentials not configured"):
        provider.search(
            bbox=(108.0, 15.0, 109.0, 16.0),
            date_range=("2024-01-01", "2024-01-02"),
        )

    with pytest.raises(ProviderAuthError, match="CDSE credentials not configured"):
        provider._get_token()


def test_planetary_healthcheck_contract() -> None:
    provider = PlanetaryComputerProvider()
    health = provider.health_check()
    assert isinstance(health, dict)
    assert "configured" in health
    assert "reachable" in health
    assert "authenticated" in health
    assert "latency_ms" in health
    assert "last_success" in health
    assert "data_freshness" in health
    assert isinstance(health["configured"], bool)
    assert isinstance(health["reachable"], bool)
    assert isinstance(health["authenticated"], bool)
    assert health["latency_ms"] is None or isinstance(health["latency_ms"], (int, float))


def test_status_endpoint_returns_live_provider_health(tmp_path: Path) -> None:
    db_file = tmp_path / "test_status.db"
    audit_file = tmp_path / "test_status_audit.jsonl"
    with patch("src.api.server.DB_PATH", db_file), \
         patch("src.api.server.AUDIT_LOG_PATH", audit_file), \
         patch("src.api.server._seed_alerts_from_geojson"):
        init_db()
        with TestClient(app) as client:
            resp = client.get("/api/status")
            assert resp.status_code == 200
            res = resp.json()
            assert res["status"] == "SUCCESS"
            data = res["data"]
            assert "status" in data
            assert data["status"] in ("OPERATIONAL", "DEGRADED")
            assert "providers" in data
            providers = data["providers"]
            assert "planetary_computer" in providers
            assert "cdse" in providers
            for name, health in providers.items():
                assert "configured" in health
                assert "reachable" in health
                assert "authenticated" in health
                assert "latency_ms" in health
                assert "last_success" in health
                assert "data_freshness" in health


def test_status_endpoint_degraded_when_provider_fails(tmp_path: Path) -> None:
    db_file = tmp_path / "test_status_fail.db"
    audit_file = tmp_path / "test_status_fail_audit.jsonl"
    failing_health = {
        "planetary_computer": {
            "configured": True,
            "reachable": False,
            "authenticated": False,
            "latency_ms": None,
            "last_success": None,
            "data_freshness": None,
        },
        "cdse": {
            "configured": False,
            "reachable": False,
            "authenticated": False,
            "latency_ms": None,
            "last_success": None,
            "data_freshness": None,
        },
    }
    with patch("src.api.server.DB_PATH", db_file), \
         patch("src.api.server.AUDIT_LOG_PATH", audit_file), \
         patch("src.api.server._seed_alerts_from_geojson"), \
         patch("src.api.server.check_provider_health", return_value=failing_health):
        init_db()
        with TestClient(app) as client:
            resp = client.get("/api/status")
            assert resp.status_code == 200
            res = resp.json()
            assert res["status"] == "SUCCESS"
            data = res["data"]
            assert data["status"] == "DEGRADED"
            assert data["services"]["planetary_computer"] == "DOWN"
            assert data["services"]["cdse_copernicus"] == "DOWN"


def test_factory_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    with pytest.raises(RuntimeError):
        get_provider("mock")

    monkeypatch.setenv("APP_ENV", "test")
    mock_prov = get_provider("mock")
    assert isinstance(mock_prov, MockSatelliteProvider)

    pc_prov = get_provider("planetary_computer")
    assert isinstance(pc_prov, PlanetaryComputerProvider)

    cdse_prov = get_provider("cdse")
    assert isinstance(cdse_prov, CdseProvider)

    with pytest.raises(ProviderNotAvailableError):
        get_provider("unsupported_provider")


def test_planetary_search_network_failure() -> None:
    provider = PlanetaryComputerProvider(endpoint="https://invalid.endpoint.local.dev")
    with pytest.raises(ProviderUnreachableError):
        provider.search(
            bbox=(108.0, 15.0, 109.0, 16.0),
            date_range=("2024-01-01", "2024-01-02"),
        )
