import json
from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from src.api.auth import Role, create_access_token
from src.api.server import app, init_db, get_db_connection


@pytest.fixture
def auth_headers():
    token = create_access_token(user_id="op-01", role=Role.OPERATOR)
    return {"Authorization": f"Bearer {token}"}


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
    res = resp.json()
    assert res["status"] == "SUCCESS"
    data = res["data"]
    assert data.get("type") == "FeatureCollection"
    assert "features" in data


def test_get_alerts(client: TestClient):
    resp = client.get("/api/alerts")
    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "SUCCESS"
    alerts = res["data"]
    assert isinstance(alerts, list)
    assert len(alerts) >= 1
    assert alerts[0]["alert_id"] == "ALT_TEST_001"
    assert alerts[0]["level"] == "CRITICAL"


def test_get_alerts_filter(client: TestClient):
    resp_crit = client.get("/api/alerts?level=CRITICAL")
    assert resp_crit.status_code == 200
    crit_res = resp_crit.json()
    assert crit_res["status"] == "SUCCESS"
    assert len(crit_res["data"]) == 1

    resp_high = client.get("/api/alerts?level=HIGH")
    assert resp_high.status_code == 200
    high_res = resp_high.json()
    assert high_res["status"] == "SUCCESS"
    assert len(high_res["data"]) == 0


def test_audit_alert_success(client: TestClient, auth_headers):
    resp = client.post(
        "/api/alerts/ALT_TEST_001/audit",
        json={"status": "CONFIRMED", "note": "Verified by operator"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "SUCCESS"
    assert res["data"]["status"] == "CONFIRMED"

    resp_after = client.get("/api/alerts")
    assert resp_after.json()["data"][0]["status"] == "CONFIRMED"


def test_audit_alert_invalid_status(client: TestClient, auth_headers):
    resp = client.post(
        "/api/alerts/ALT_TEST_001/audit",
        json={"status": "INVALID_STATUS"},
        headers=auth_headers,
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
    res = resp.json()
    assert res["status"] == "SUCCESS"
    data = res["data"]
    assert data["status"] == "OPERATIONAL"
    assert "metrics" in data


def test_scan_map_aoi_success(client: TestClient, auth_headers):
    payload = {
        "bbox": [108.15, 16.05, 108.25, 16.15],
        "source": "satellite_hd",
        "date": "2024-06-01",
    }
    resp = client.post("/api/scan-map-aoi", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "SUCCESS"
    data = res["data"]
    assert data["status"] == "SUCCESS"
    assert data["source"] == "satellite_hd"
    assert data["tiles_fetched"] >= 1
    assert data["detections_count"] >= 1
    assert len(data["alerts"]) >= 1

    alerts_resp = client.get("/api/alerts")
    assert alerts_resp.status_code == 200
    all_alerts = alerts_resp.json()["data"]
    assert len(all_alerts) >= 2


def test_scan_map_aoi_invalid_bbox(client: TestClient, auth_headers):
    resp = client.post("/api/scan-map-aoi", json={"bbox": [108.0, 16.0]}, headers=auth_headers)
    assert resp.status_code == 400


def test_scan_map_aoi_sources(client: TestClient, auth_headers):
    payload_gibs = {
        "bbox": [108.0, 16.0, 109.0, 17.0],
        "source": "nasa_gibs",
        "date": "2024-05-01",
    }
    resp_gibs = client.post("/api/scan-map-aoi", json=payload_gibs, headers=auth_headers)
    assert resp_gibs.status_code == 200
    assert resp_gibs.json()["data"]["source"] == "nasa_gibs"

    payload_s2 = {
        "bbox": [108.1, 16.1, 108.2, 16.2],
        "source": "sentinel2",
    }
    resp_s2 = client.post("/api/scan-map-aoi", json=payload_s2, headers=auth_headers)
    assert resp_s2.status_code == 200
    assert resp_s2.json()["data"]["source"] == "sentinel2"


def test_get_alert_by_id_v2_contracts(client: TestClient):
    resp = client.get("/api/alerts/ALT_TEST_001")
    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "SUCCESS"
    data = res["data"]
    assert data["alert_id"] == "ALT_TEST_001"
    assert "hypothesis" in data
    assert "statement" in data["hypothesis"]
    assert "primary_explanation" in data["hypothesis"]
    assert "detailed_rationale" in data["hypothesis"]

    assert "confidence" in data
    assert data["confidence"]["calibrated_probability"] is None
    assert data["confidence"]["calibration_state"] == "NOT_CALIBRATED"
    assert "quality_score" in data["confidence"]

    assert "evidence" in data
    assert "supporting" in data["evidence"]
    assert len(data["evidence"]["supporting"]) >= 1
    assert "refuting" in data["evidence"]

    assert "chips" in data
    assert "preview" in data["chips"]
    assert "analysis" in data["chips"]
    assert "raw" in data["chips"]

    resp_not_found = client.get("/api/alerts/NONEXISTENT_ID")
    assert resp_not_found.status_code == 404


def test_alert_action_endpoint_tamper_resistant_hash(client: TestClient, auth_headers):
    resp = client.post(
        "/api/alerts/ALT_TEST_001/action",
        json={"action": "FLAG_UNCERTAIN", "operator_id": "op_alpha", "notes": "Suspected decoy"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "SUCCESS"
    data = res["data"]
    assert data["alert_id"] == "ALT_TEST_001"
    assert data["status"] == "FLAGGED_UNCERTAIN"
    assert data["action"] == "FLAG_UNCERTAIN"
    assert "entry_hash" in data
    assert len(data["entry_hash"]) == 64
    assert data["previous_hash"] == "GENESIS"

    resp_list = client.get("/api/alerts")
    assert resp_list.status_code == 200
    assert resp_list.json()["data"][0]["status"] == "FLAGGED_UNCERTAIN"


def test_status_endpoint_returns_capabilities(client: TestClient):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "SUCCESS"
    data = res["data"]
    assert "capabilities" in data
    caps = data["capabilities"]
    assert caps["sar_vessel_detector"] == "IMPLEMENTED"
    assert caps["review_audit_store"] == "IMPLEMENTED"
    assert caps["alert_package_builder"] == "IMPLEMENTED"
    assert caps["hypothesis_engine"] == "IMPLEMENTED"
    assert "providers" in caps
    assert caps["providers"]["planetary_computer"] == "IMPLEMENTED"
    assert caps["providers"]["cdse"] == "IMPLEMENTED"
    assert caps["providers"]["mock"] == "MOCK"


def test_multitier_chips_endpoint(client: TestClient):
    from src.api.server import OUTPUTS_DIR
    chip_dir = OUTPUTS_DIR / "chips"
    chip_dir.mkdir(parents=True, exist_ok=True)
    prev_file = chip_dir / "preview_test.png"
    prev_file.write_bytes(b"\x89PNG\r\n\x1a\n")
    raw_file = chip_dir / "raw_test.tif"
    raw_file.write_bytes(b"II*\x00")
    try:
        resp_prev = client.get("/api/chips/preview_test.png")
        assert resp_prev.status_code == 200
        assert resp_prev.headers["content-type"] == "image/png"

        resp_raw = client.get("/api/chips/raw_test.tif")
        assert resp_raw.status_code == 200
        assert resp_raw.headers["content-type"] == "image/tiff"
    finally:
        if prev_file.exists():
            prev_file.unlink()
        if raw_file.exists():
            raw_file.unlink()
