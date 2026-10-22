from src.ingestion.cdse_client import CopernicusClient


def test_copernicus_client_mock_search():
    client = CopernicusClient(mock_mode=True)
    results = client.search_sentinel1(
        bbox=(108.0, 15.0, 110.0, 17.0),
        date_range=("2026-10-01", "2026-10-22"),
        product_type="GRD",
        limit=5,
    )
    assert len(results) == 3
    first = results[0]
    assert first["type"] == "Feature"
    assert first["collection"] == "SENTINEL-1"
    assert "geometry" in first
    assert "properties" in first
    assert first["properties"]["productType"] == "GRD"
    assert "download" in first["assets"]
