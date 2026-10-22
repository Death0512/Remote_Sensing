from src.ingestion.cdse_client import CopernicusClient
from src.ingestion.gfw_client import GfwClient
from src.ingestion.planetary_client import PlanetaryComputerClient


def test_planetary_computer_client_search():
    client = PlanetaryComputerClient(mock_mode=True)
    bbox = (107.0, 15.0, 109.0, 17.0)
    date_range = ("2026-10-01", "2026-10-22")

    results = client.search_scenes(bbox=bbox, date_range=date_range, limit=5)
    assert len(results) > 0
    for scene in results:
        assert "id" in scene
        assert "geometry" in scene
        assert "assets" in scene
        assert scene["collection"] == "sentinel-1-rtc"


def test_copernicus_client_search():
    client = CopernicusClient(mock_mode=True)
    bbox = (107.0, 15.0, 109.0, 17.0)
    date_range = ("2026-10-01", "2026-10-22")

    results = client.search_sentinel1(bbox=bbox, date_range=date_range, limit=5)
    assert len(results) > 0
    for scene in results:
        assert "id" in scene
        assert scene["collection"] == "SENTINEL-1"
        assert scene["properties"]["productType"] == "GRD"


def test_gfw_client_query():
    client = GfwClient(mock_mode=True)
    bbox = (107.0, 15.0, 109.0, 17.0)
    date_range = ("2026-10-01", "2026-10-22")

    results = client.get_vessel_presence(bbox=bbox, date_range=date_range)
    assert len(results) > 0
    for entry in results:
        assert "mmsi" in entry
        assert "lat" in entry
        assert "lon" in entry
        assert "confidence" in entry


def test_live_clients_return_empty_on_failure():
    bbox = (107.0, 15.0, 109.0, 17.0)
    date_range = ("2026-10-01", "2026-10-22")

    pc_client = PlanetaryComputerClient(endpoint="http://127.0.0.1:9", mock_mode=False)
    assert pc_client.search_scenes(bbox=bbox, date_range=date_range) == []

    cdse_client = CopernicusClient(endpoint="http://127.0.0.1:9", mock_mode=False)
    assert cdse_client.search_sentinel1(bbox=bbox, date_range=date_range) == []

    gfw_client = GfwClient(endpoint="http://127.0.0.1:9", mock_mode=False)
    assert gfw_client.get_vessel_presence(bbox=bbox, date_range=date_range) == []

