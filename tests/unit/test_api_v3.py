from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import pytest
from fastapi.testclient import TestClient

from src.api.auth import Role, create_access_token
from src.api.server import ALLOWED_ORIGINS, app, get_db_connection, init_db


@pytest.fixture(autouse=True)
def setup_api_database(tmp_path, monkeypatch):
    test_db = tmp_path / "test_alerts.db"
    test_audit = tmp_path / "test_audit.jsonl"
    monkeypatch.setattr("src.api.server.DB_PATH", test_db)
    monkeypatch.setattr("src.api.server.AUDIT_LOG_PATH", test_audit)
    init_db()

    with sqlite3.connect(str(test_db)) as conn:
        conn.execute("DELETE FROM alerts")
        conn.execute(
            """
            INSERT INTO alerts (
                alert_id, det_id, level, zone, score, target_class,
                sensor, speed_knots, heading_deg, is_dark, lat, lon,
                chip_path, timestamp, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "ALT_TEST_001",
                "DET_001",
                "CRITICAL",
                1,
                0.92,
                "vessel",
                "S1_GRD",
                14.2,
                120.0,
                1,
                16.12,
                108.25,
                "ALT_TEST_001.png",
                datetime.now(timezone.utc).isoformat(),
                "PENDING",
            ),
        )
        conn.commit()
    yield


@pytest.fixture
def client():
    return TestClient(app)


def test_cors_origin_whitelist(client):
    allowed_origin = ALLOWED_ORIGINS[0]
    resp = client.options(
        "/api/status",
        headers={
            "Origin": allowed_origin,
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.headers.get("access-control-allow-origin") == allowed_origin

    disallowed_origin = "http://malicious-site.com"
    resp_bad = client.options(
        "/api/status",
        headers={
            "Origin": disallowed_origin,
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp_bad.headers.get("access-control-allow-origin") != disallowed_origin


def test_auth_login_token(client):
    resp = client.post("/api/auth/token", json={"username": "operator", "password": "operator123"})
    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "SUCCESS"
    assert "access_token" in res["data"]
    assert res["data"]["user"]["role"] == "operator"

    bad_resp = client.post("/api/auth/token", json={"username": "operator", "password": "wrong"})
    assert bad_resp.status_code == 401
    bad_res = bad_resp.json()
    assert bad_res["status"] == "ERROR"
    assert bad_res["errors"][0]["code"] == "UNAUTHORIZED"


def test_rbac_review_action_permissions(client):
    viewer_token = create_access_token(user_id="view-01", role=Role.VIEWER)
    operator_token = create_access_token(user_id="op-01", role=Role.OPERATOR)

    unauth_resp = client.post(
        "/api/alerts/ALT_TEST_001/action",
        json={"action": "CONFIRM"},
    )
    assert unauth_resp.status_code == 401
    unauth_res = unauth_resp.json()
    assert unauth_res["status"] == "ERROR"
    assert unauth_res["errors"][0]["code"] == "UNAUTHORIZED"

    viewer_resp = client.post(
        "/api/alerts/ALT_TEST_001/action",
        json={"action": "CONFIRM"},
        headers={"Authorization": f"Bearer {viewer_token}"},
    )
    assert viewer_resp.status_code == 403
    viewer_res = viewer_resp.json()
    assert viewer_res["status"] == "ERROR"
    assert viewer_res["errors"][0]["code"] == "FORBIDDEN"

    op_resp = client.post(
        "/api/alerts/ALT_TEST_001/action",
        json={"action": "CONFIRM", "note": "Target confirmed by op-01"},
        headers={"Authorization": f"Bearer {operator_token}"},
    )
    assert op_resp.status_code == 200
    op_res = op_resp.json()
    assert op_res["status"] == "SUCCESS"
    assert op_res["data"]["status"] == "CONFIRMED"
    assert op_res["data"]["operator_id"] == "op-01"
    assert op_res["provenance"]["operator_id"] == "op-01"


def test_audit_endpoint_operator_id(client):
    operator_token = create_access_token(user_id="op-01", role=Role.OPERATOR)
    resp = client.post(
        "/api/alerts/ALT_TEST_001/audit",
        json={"status": "CONFIRM", "note": "Verified contact"},
        headers={"Authorization": f"Bearer {operator_token}"},
    )
    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "SUCCESS"
    assert res["data"]["operator_id"] == "op-01"
    assert res["provenance"]["operator_id"] == "op-01"


def test_structured_envelopes_alerts_and_status(client):
    status_resp = client.get("/api/status")
    assert status_resp.status_code == 200
    s_res = status_resp.json()
    assert s_res["status"] == "SUCCESS"
    assert "metrics" in s_res["data"]
    assert "capabilities" in s_res["data"]
    assert s_res["provenance"]["pipeline_version"] == "3.0.0"

    alerts_resp = client.get("/api/alerts")
    assert alerts_resp.status_code == 200
    a_res = alerts_resp.json()
    assert a_res["status"] == "SUCCESS"
    assert isinstance(a_res["data"], list)
    assert len(a_res["data"]) >= 1
    found_alert = next((a for a in a_res["data"] if a["alert_id"] == "ALT_TEST_001"), None)
    assert found_alert is not None
    assert "hypothesis" in found_alert
    assert "confidence" in found_alert

    single_resp = client.get("/api/alerts/ALT_TEST_001")
    assert single_resp.status_code == 200
    single_res = single_resp.json()
    assert single_res["status"] == "SUCCESS"
    assert single_res["data"]["alert_id"] == "ALT_TEST_001"
    assert single_res["data"]["hypothesis"]["score"] == 0.92

    not_found = client.get("/api/alerts/ALT_NON_EXISTENT")
    assert not_found.status_code == 404
    nf_res = not_found.json()
    assert nf_res["status"] == "ERROR"
    assert nf_res["errors"][0]["code"] == "NOT_FOUND"


def test_scan_aoi_rbac_and_data_unavailable(client, monkeypatch):
    viewer_token = create_access_token(user_id="view-01", role=Role.VIEWER)
    analyst_token = create_access_token(user_id="an-01", role=Role.ANALYST)

    viewer_resp = client.post(
        "/api/scan-map-aoi",
        json={"bbox": [108.1, 16.0, 108.3, 16.2], "source": "satellite_hd"},
        headers={"Authorization": f"Bearer {viewer_token}"},
    )
    assert viewer_resp.status_code == 403

    def mock_fetch_fail(source, date_str, zoom, x, y):
        from src.ingestion.exceptions import DataUnavailableError
        raise DataUnavailableError("Simulated tile imagery network timeout")

    monkeypatch.setattr("src.api.server._fetch_tile", mock_fetch_fail)

    analyst_resp = client.post(
        "/api/scan-map-aoi",
        json={"bbox": [108.1, 16.0, 108.3, 16.2], "source": "satellite_hd"},
        headers={"Authorization": f"Bearer {analyst_token}"},
    )
    assert analyst_resp.status_code == 503
    res = analyst_resp.json()
    assert res["status"] == "DATA_UNAVAILABLE"
    assert res["errors"][0]["code"] == "DATA_UNAVAILABLE"


def test_chip_serving_and_traversal_protection(client, tmp_path, monkeypatch):
    chip_dir = tmp_path / "chips"
    chip_dir.mkdir(parents=True, exist_ok=True)
    sample_png = chip_dir / "sample.png"
    sample_png.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")
    monkeypatch.setattr("src.api.server.OUTPUTS_DIR", tmp_path)

    resp = client.get("/api/chips/sample.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"

    traversal_resp = client.get("/api/chips/../../etc/passwd")
    assert traversal_resp.status_code == 404


def test_production_environment_missing_jwt_secret_fails_closed(monkeypatch):
    import subprocess
    import sys
    from src.api.auth import get_jwt_secret

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="JWT_SECRET must be explicitly set and secure in production"):
        get_jwt_secret()

    monkeypatch.setenv("JWT_SECRET", "super-secret-tactical-defense-key-2026")
    with pytest.raises(RuntimeError, match="JWT_SECRET must be explicitly set and secure in production"):
        get_jwt_secret()

    monkeypatch.setenv("JWT_SECRET", "dev-secret-key-change-in-production")
    with pytest.raises(RuntimeError, match="JWT_SECRET must be explicitly set and secure in production"):
        get_jwt_secret()

    monkeypatch.setenv("JWT_SECRET", "valid-production-secret-of-sufficient-length-here")
    assert get_jwt_secret() == "valid-production-secret-of-sufficient-length-here"

    proc = subprocess.run(
        [sys.executable, "-c", "import os; os.environ['ENVIRONMENT']='production'; os.environ.pop('JWT_SECRET', None); import src.api.auth"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "JWT_SECRET must be explicitly set and secure in production" in proc.stderr

    proc_default = subprocess.run(
        [sys.executable, "-c", "import os; os.environ['ENVIRONMENT']='production'; os.environ['JWT_SECRET']='secret'; import src.api.auth"],
        capture_output=True,
        text=True,
    )
    assert proc_default.returncode != 0
    assert "JWT_SECRET must be explicitly set and secure in production" in proc_default.stderr


def test_demo_credentials_disabled_in_production(monkeypatch):
    from src.api.auth import authenticate_user, get_demo_users

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET", "valid-production-secret-of-sufficient-length-here")
    assert get_demo_users() == {}
    assert authenticate_user("operator", "operator123") is None


def test_token_rejected_at_expiry_boundary():
    import base64
    from datetime import datetime, timezone, timedelta
    import hmac
    import hashlib
    from src.api.auth import Role, create_access_token, decode_access_token, get_jwt_secret

    token = create_access_token(
        user_id="op-01",
        role=Role.OPERATOR,
        expires_delta=timedelta(seconds=0),
    )
    with pytest.raises(ValueError, match="Token has expired"):
        decode_access_token(token)

    now_ts = int(datetime.now(timezone.utc).timestamp())
    active_token = create_access_token(
        user_id="op-01",
        role=Role.OPERATOR,
        expires_delta=timedelta(seconds=3600),
    )
    parts = active_token.split(".")
    payload_dict = json.loads(base64.urlsafe_b64decode(parts[1] + "==").decode("utf-8"))
    payload_dict["exp"] = now_ts
    encoded_p = base64.urlsafe_b64encode(json.dumps(payload_dict, separators=(",", ":")).encode("utf-8")).rstrip(b"=").decode("utf-8")
    signing_input = f"{parts[0]}.{encoded_p}".encode("utf-8")
    sig = base64.urlsafe_b64encode(hmac.new(get_jwt_secret().encode("utf-8"), signing_input, hashlib.sha256).digest()).rstrip(b"=").decode("utf-8")
    boundary_token = f"{parts[0]}.{encoded_p}.{sig}"
    with pytest.raises(ValueError, match="Token has expired"):
        decode_access_token(boundary_token)


def test_xss_payload_escaped_in_dashboard(client):
    import subprocess
    xss_alert_id = "ALT_XSS_INJ"
    xss_target = "<img src=x onerror=alert('target')>"
    xss_statement = "<script>alert('statement')</script>"
    xss_rationale = "<svg/onload=alert('rationale')>"
    xss_evidence_kind = "<b>KIND</b>"
    xss_evidence_ref = "<iframe src='alert(1)'>"

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO alerts (
                alert_id, det_id, level, zone, score, target_class,
                sensor, speed_knots, heading_deg, is_dark, lat, lon,
                chip_path, timestamp, status, hypothesis_json, evidence_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                xss_alert_id,
                "DET_XSS",
                "CRITICAL",
                1,
                0.99,
                xss_target,
                "SAR",
                10.0,
                90.0,
                0,
                16.0,
                108.0,
                "xss.png",
                datetime.now(timezone.utc).isoformat(),
                "PENDING",
                json.dumps({
                    "statement": xss_statement,
                    "primary_explanation": "<script>alert('explanation')</script>",
                    "detailed_rationale": xss_rationale,
                }),
                json.dumps({
                    "supporting": [{
                        "kind": xss_evidence_kind,
                        "value": 1.0,
                        "source": "SAR",
                        "supports": "VESSEL",
                        "references": [xss_evidence_ref],
                    }],
                    "refuting": [],
                }),
            ),
        )
        conn.commit()

    resp = client.get(f"/api/alerts/{xss_alert_id}")
    assert resp.status_code == 200
    alert_data = resp.json()["data"]
    assert alert_data["alert_id"] == xss_alert_id

    js_verifier = """
    const { escapeHtml } = require('./web/app.js');
    const alert = JSON.parse(process.argv[1]);
    const fields = [
        alert.alert_id,
        alert.target_class,
        alert.hypothesis.statement,
        alert.hypothesis.primary_explanation,
        alert.hypothesis.detailed_rationale,
        alert.evidence.supporting[0].kind,
        alert.evidence.supporting[0].references[0]
    ];
    for (const f of fields) {
        const escaped = escapeHtml(f);
        if (escaped.includes('<') || escaped.includes('>') || escaped.includes('"') || escaped.includes("'")) {
            process.exit(1);
        }
    }
    process.exit(0);
    """
    res = subprocess.run(
        ["node", "-e", js_verifier, json.dumps(alert_data)],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent.parent),
    )
    assert res.returncode == 0


def test_operator_id_in_ledger_strictly_equals_jwt_user_id(client):
    operator_token = create_access_token(user_id="authorized-operator-77", role=Role.OPERATOR)
    spoofed_payload = {
        "action": "CONFIRM",
        "operator_id": "malicious-spoofed-operator",
        "note": "Attacker trying to spoof ledger operator_id",
        "notes": "Attacker trying to spoof ledger operator_id",
    }
    resp = client.post(
        "/api/alerts/ALT_TEST_001/action",
        json=spoofed_payload,
        headers={"Authorization": f"Bearer {operator_token}"},
    )
    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "SUCCESS"
    assert res["data"]["operator_id"] == "authorized-operator-77"
    assert res["provenance"]["operator_id"] == "authorized-operator-77"

    with get_db_connection() as conn:
        cursor = conn.execute("SELECT operator_id FROM review_audit_log WHERE alert_id = ?", ("ALT_TEST_001",))
        rows = cursor.fetchall()
        assert len(rows) > 0
        for r in rows:
            assert r[0] == "authorized-operator-77"


def test_production_jwt_secret_entropy_and_length_gate(monkeypatch):
    from src.api.auth import get_jwt_secret

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET", "short-secret-under-32-chars")
    with pytest.raises(RuntimeError, match="JWT_SECRET must be at least 32 characters with sufficient entropy in production"):
        get_jwt_secret()

    monkeypatch.setenv("JWT_SECRET", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    with pytest.raises(RuntimeError, match="JWT_SECRET must be at least 32 characters with sufficient entropy in production"):
        get_jwt_secret()

    monkeypatch.setenv("JWT_SECRET", "abcdefg-abcdefg-abcdefg-abcdefg")
    with pytest.raises(RuntimeError, match="JWT_SECRET must be at least 32 characters with sufficient entropy in production"):
        get_jwt_secret()

    monkeypatch.setenv("JWT_SECRET", "valid-production-secret-of-sufficient-length-and-entropy-2026")
    assert get_jwt_secret() == "valid-production-secret-of-sufficient-length-and-entropy-2026"


def test_require_auth_for_reads_blocks_unauthenticated(client, monkeypatch):
    monkeypatch.setenv("REQUIRE_AUTH_FOR_READS", "1")

    resp_unauth = client.get("/api/alerts")
    assert resp_unauth.status_code == 401
    assert resp_unauth.json()["errors"][0]["code"] == "UNAUTHORIZED"

    resp_chip_unauth = client.get("/api/chips/sample.png")
    assert resp_chip_unauth.status_code == 401

    resp_status_unauth = client.get("/api/status")
    assert resp_status_unauth.status_code == 401
    assert resp_status_unauth.json()["errors"][0]["code"] == "UNAUTHORIZED"

    resp_zones_unauth = client.get("/api/zones")
    assert resp_zones_unauth.status_code == 401
    assert resp_zones_unauth.json()["errors"][0]["code"] == "UNAUTHORIZED"

    viewer_token = create_access_token(user_id="view-01", role=Role.VIEWER)
    resp_auth = client.get("/api/alerts", headers={"Authorization": f"Bearer {viewer_token}"})
    assert resp_auth.status_code == 200
    assert resp_auth.json()["status"] == "SUCCESS"

    resp_status = client.get("/api/status", headers={"Authorization": f"Bearer {viewer_token}"})
    assert resp_status.status_code == 200
    assert resp_status.json()["status"] == "SUCCESS"

    resp_zones = client.get("/api/zones", headers={"Authorization": f"Bearer {viewer_token}"})
    assert resp_zones.status_code == 200
    assert resp_zones.json()["status"] == "SUCCESS"


def test_token_missing_mandatory_exp_rejected():
    import base64
    import hashlib
    import hmac
    import json
    from src.api.auth import InvalidTokenError, Role, create_access_token, decode_access_token, get_jwt_secret

    token = create_access_token(user_id="op-01", role=Role.OPERATOR)
    parts = token.split(".")
    payload = json.loads(base64.urlsafe_b64decode(parts[1] + "==").decode("utf-8"))

    payload_no_exp = dict(payload)
    payload_no_exp.pop("exp", None)
    encoded_p1 = base64.urlsafe_b64encode(json.dumps(payload_no_exp, separators=(",", ":")).encode("utf-8")).rstrip(b"=").decode("utf-8")
    sig1 = base64.urlsafe_b64encode(hmac.new(get_jwt_secret().encode("utf-8"), f"{parts[0]}.{encoded_p1}".encode("utf-8"), hashlib.sha256).digest()).rstrip(b"=").decode("utf-8")
    token_no_exp = f"{parts[0]}.{encoded_p1}.{sig1}"

    with pytest.raises(InvalidTokenError, match="Token missing mandatory expiration claim 'exp'"):
        decode_access_token(token_no_exp)

    payload_none_exp = dict(payload)
    payload_none_exp["exp"] = None
    encoded_p2 = base64.urlsafe_b64encode(json.dumps(payload_none_exp, separators=(",", ":")).encode("utf-8")).rstrip(b"=").decode("utf-8")
    sig2 = base64.urlsafe_b64encode(hmac.new(get_jwt_secret().encode("utf-8"), f"{parts[0]}.{encoded_p2}".encode("utf-8"), hashlib.sha256).digest()).rstrip(b"=").decode("utf-8")
    token_none_exp = f"{parts[0]}.{encoded_p2}.{sig2}"

    with pytest.raises(InvalidTokenError, match="Token missing mandatory expiration claim 'exp'"):
        decode_access_token(token_none_exp)


def test_insecure_secret_with_surrounding_whitespace_rejected(monkeypatch):
    from src.api.auth import get_jwt_secret

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET", "  dev-secret-key-change-in-production  ")
    with pytest.raises(RuntimeError, match="JWT_SECRET must be explicitly set and secure in production"):
        get_jwt_secret()

    monkeypatch.setenv("JWT_SECRET", "\tsuper-secret-tactical-defense-key-2026\n")
    with pytest.raises(RuntimeError, match="JWT_SECRET must be explicitly set and secure in production"):
        get_jwt_secret()

    monkeypatch.setenv("JWT_SECRET", "   secret   ")
    with pytest.raises(RuntimeError, match="JWT_SECRET must be explicitly set and secure in production"):
        get_jwt_secret()

    valid_secret = "   valid-production-secret-of-sufficient-length-and-entropy-2026   "
    monkeypatch.setenv("JWT_SECRET", valid_secret)
    assert get_jwt_secret() == valid_secret.strip()


def test_general_500_exception_envelope(monkeypatch):
    from fastapi.testclient import TestClient
    unhandled_client = TestClient(app, raise_server_exceptions=False)

    def mock_db_fail():
        raise RuntimeError("Database connection suddenly dropped")

    monkeypatch.setattr("src.api.server.get_db_connection", mock_db_fail)
    resp = unhandled_client.get("/api/alerts")
    assert resp.status_code == 500
    res = resp.json()
    assert res["status"] == "ERROR"
    assert res["data"] is None
    assert res["provenance"] is None
    assert len(res["errors"]) == 1
    assert res["errors"][0]["code"] == "INTERNAL_SERVER_ERROR"
    assert "Database connection suddenly dropped" in res["errors"][0]["message"]
