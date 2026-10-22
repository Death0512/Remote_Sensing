import json
from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from src.api.server import app, init_db, get_db_connection


@pytest.fixture
def client(tmp_path: Path):
    db_file = tmp_path / "test_alerts.db"
    audit_file = tmp_path / "test_audit.jsonl"
    with patch("src.api.server.DB_PATH", db_file), \
         patch("src.api.server.AUDIT_LOG_PATH", audit_file), \
         patch("src.api.server._seed_alerts_from_geojson"):
        init_db()
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO alerts (
                    alert_id, det_id, level, zone, score, target_class,
                    sensor, speed_knots, heading_deg, is_dark, lat, lon, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "ALT_TEST_001",
                    "DET_001",
                    "CRITICAL",
                    1,
                    88.5,
                    "vessel",
                    "S1_GRD",
                    15.2,
                    145.0,
                    1,
                    16.2,
                    108.5,
                    "PENDING",
                ),
            )
            conn.commit()

        with TestClient(app) as test_client:
            yield test_client


def test_get_zones(client: TestClient):
    resp = client.get("/api/zones")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("type") == "FeatureCollection"
    assert "features" in data


def test_get_alerts(client: TestClient):
    resp = client.get("/api/alerts")
    assert resp.status_code == 200
    alerts = resp.json()
    assert isinstance(alerts, list)
    assert len(alerts) >= 1
    assert alerts[0]["alert_id"] == "ALT_TEST_001"
    assert alerts[0]["level"] == "CRITICAL"


def test_get_alerts_filter(client: TestClient):
    resp_crit = client.get("/api/alerts?level=CRITICAL")
    assert resp_crit.status_code == 200
    assert len(resp_crit.json()) == 1

    resp_high = client.get("/api/alerts?level=HIGH")
    assert resp_high.status_code == 200
    assert len(resp_high.json()) == 0


def test_audit_alert_success(client: TestClient):
    resp = client.post(
        "/api/alerts/ALT_TEST_001/audit",
        json={"status": "CONFIRMED", "note": "Verified by operator"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "CONFIRMED"

    resp_after = client.get("/api/alerts")
    assert resp_after.json()[0]["status"] == "CONFIRMED"


def test_audit_alert_invalid_status(client: TestClient):
    resp = client.post(
        "/api/alerts/ALT_TEST_001/audit",
        json={"status": "INVALID_STATUS"},
    )
    assert resp.status_code == 400


def test_get_chip(client: TestClient, tmp_path: Path):
    from src.api.server import OUTPUTS_DIR
    chip_dir = OUTPUTS_DIR / "chips"
    chip_dir.mkdir(parents=True, exist_ok=True)
    chip_file = chip_dir / "ALT_TEST_001.png"
    chip_file.write_bytes(b"\x89PNG\r\n\x1a\n")
    try:
        resp = client.get("/api/chips/ALT_TEST_001")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

        resp_not_found = client.get("/api/chips/NONEXISTENT")
        assert resp_not_found.status_code == 404

        resp_traversal = client.get("/api/chips/../../etc/passwd")
        assert resp_traversal.status_code == 404
    finally:
        if chip_file.exists():
            chip_file.unlink()


def test_get_status(client: TestClient):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "OPERATIONAL"
    assert "metrics" in data


def test_scan_map_aoi_success(client: TestClient):
    payload = {
        "bbox": [108.15, 16.05, 108.25, 16.15],
        "source": "satellite_hd",
        "date": "2024-06-01",
    }
    resp = client.post("/api/scan-map-aoi", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert data["source"] == "satellite_hd"
    assert data["tiles_fetched"] >= 1
    assert data["detections_count"] >= 1
    assert len(data["alerts"]) >= 1

    alerts_resp = client.get("/api/alerts")
    assert alerts_resp.status_code == 200
    all_alerts = alerts_resp.json()
    assert len(all_alerts) >= 2


def test_scan_map_aoi_invalid_bbox(client: TestClient):
    resp = client.post("/api/scan-map-aoi", json={"bbox": [108.0, 16.0]})
    assert resp.status_code == 400


def test_scan_map_aoi_sources(client: TestClient):
    payload_gibs = {
        "bbox": [108.0, 16.0, 109.0, 17.0],
        "source": "nasa_gibs",
        "date": "2024-05-01",
    }
    resp_gibs = client.post("/api/scan-map-aoi", json=payload_gibs)
    assert resp_gibs.status_code == 200
    assert resp_gibs.json()["source"] == "nasa_gibs"

    payload_s2 = {
        "bbox": [108.1, 16.1, 108.2, 16.2],
        "source": "sentinel2",
    }
    resp_s2 = client.post("/api/scan-map-aoi", json=payload_s2)
    assert resp_s2.status_code == 200
    assert resp_s2.json()["source"] == "sentinel2"

