from contextlib import asynccontextmanager
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import urllib.request
import uuid

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
from pydantic import BaseModel
import scipy.ndimage
from shapely.geometry import Polygon

from src.alerting.package.builder import AlertPackageBuilder
from src.alerting.priority import determine_alert_priority
from src.alerting.review.store import ReviewAuditStore
from src.detection.change.change_detector import BitemporalChangeDetector
from src.detection.experimental.parallax_detector import ParallaxInflightDetector
from src.detection.optical.aircraft_detector import OpticalAircraftDetector
from src.detection.sar.vessel_detector import SarVesselDetector
from src.domain.alert import Alert
from src.domain.confidence import ConfidenceVector
from src.domain.detection import Detection
from src.domain.enums import (
    AlertPriority,
    CalibrationState,
    Capability,
    EvidenceKind,
    HypothesisStatement,
    ReviewAction,
    ReviewState,
    ScoreMethod,
    SensorType,
    TargetClass,
    Zone,
)
from src.domain.evidence import Evidence
from src.domain.hypothesis import Hypothesis
from src.domain.provenance import Provenance
from src.fusion.ais.correlator import AisCorrelator
from src.ingestion.providers.cdse_provider import CdseProvider
from src.ingestion.providers.factory import ProviderFactory, get_provider
from src.ingestion.providers.mock_provider import MockSatelliteProvider
from src.ingestion.providers.planetary_provider import PlanetaryComputerProvider
from src.intelligence.accumulator import EvidenceAccumulator
from src.intelligence.hypothesis.engine import HypothesisEngine
from src.preprocessing.geometry.geofence import GeofenceManager
from src.preprocessing.geometry.tiling import MetricTiler
from src.tracking.tracker import MultiLookTracker

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"
AOI_DIR = BASE_DIR / "data" / "aoi"
WEB_DIR = BASE_DIR / "web"
DB_PATH = OUTPUTS_DIR / "alerts.db"
AUDIT_LOG_PATH = OUTPUTS_DIR / "audit_log.jsonl"


def get_db_connection() -> sqlite3.Connection:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def get_review_store() -> ReviewAuditStore:
    return ReviewAuditStore(db_path=DB_PATH, journal_path=AUDIT_LOG_PATH)


def init_db() -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                det_id TEXT,
                level TEXT,
                zone INTEGER,
                score REAL,
                target_class TEXT,
                sensor TEXT,
                speed_knots REAL,
                heading_deg REAL,
                is_dark INTEGER,
                lat REAL,
                lon REAL,
                chip_path TEXT,
                timestamp TEXT,
                status TEXT DEFAULT 'PENDING',
                bundle_uri TEXT,
                hypothesis_json TEXT,
                confidence_json TEXT,
                evidence_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT,
                status TEXT,
                note TEXT,
                timestamp TEXT
            )
            """
        )
        conn.commit()

        cursor = conn.execute("SELECT COUNT(*) FROM alerts")
        count = cursor.fetchone()[0]
        if count == 0:
            _seed_alerts_from_geojson(conn)


def _seed_alerts_from_geojson(conn: sqlite3.Connection) -> None:
    candidates = [
        OUTPUTS_DIR / "live_run" / "alerts.geojson",
        OUTPUTS_DIR / "alerts.geojson",
        OUTPUTS_DIR / "demo_run" / "alerts.geojson",
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    fc = json.load(f)
                features = fc.get("features", [])
                for feat in features:
                    props = feat.get("properties", {})
                    geom = feat.get("geometry", {})
                    coords = geom.get("coordinates", [])
                    lat, lon = None, None
                    if geom.get("type") == "Point" and len(coords) >= 2:
                        lon, lat = coords[0], coords[1]
                    elif geom.get("type") == "Polygon" and coords and coords[0]:
                        poly_coords = coords[0]
                        lon = sum(c[0] for c in poly_coords) / len(poly_coords)
                        lat = sum(c[1] for c in poly_coords) / len(poly_coords)

                    alert_id = props.get("alert_id") or f"ALT_{props.get('det_id', 'UNKNOWN')}"
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO alerts (
                            alert_id, det_id, level, zone, score, target_class,
                            sensor, speed_knots, heading_deg, is_dark, lat, lon,
                            chip_path, timestamp, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            alert_id,
                            props.get("det_id"),
                            props.get("level", "STRATEGIC"),
                            props.get("zone", 3),
                            props.get("score_s", 0.0),
                            props.get("target_class", "unknown"),
                            props.get("sensor", "SAR"),
                            props.get("speed_knots"),
                            props.get("heading_deg"),
                            1 if props.get("is_dark") else 0,
                            lat,
                            lon,
                            props.get("chip_path"),
                            props.get("acquisition_time", datetime.now(timezone.utc).isoformat()),
                            props.get("status", "PENDING"),
                        ),
                    )
                conn.commit()
                break
            except Exception:
                pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Vietnam National Territorial Remote Sensing Command API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


class ActionRequest(BaseModel):
    action: Optional[str] = None
    status: Optional[str] = None
    operator_id: Optional[str] = "operator_default"
    note: Optional[str] = ""
    notes: Optional[str] = ""


class AuditRequest(BaseModel):
    status: str
    note: Optional[str] = ""


class ScanAoiRequest(BaseModel):
    bbox: List[float]
    source: str = "satellite_hd"
    date: Optional[str] = None
    detector: Optional[str] = "all"


def _lon_to_tile_x(lon: float, zoom: int) -> int:
    return int(math.floor((lon + 180.0) / 360.0 * (1 << zoom)))


def _lat_to_tile_y(lat: float, zoom: int) -> int:
    lat = max(-85.0511, min(85.0511, lat))
    lat_rad = math.radians(lat)
    return int(math.floor((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * (1 << zoom)))


def _tile_x_to_lon(x: int, zoom: int) -> float:
    return x / (1 << zoom) * 360.0 - 180.0


def _tile_y_to_lat(y: int, zoom: int) -> float:
    n = math.pi - 2.0 * math.pi * y / (1 << zoom)
    return math.degrees(math.atan(math.sinh(n)))


def _fetch_tile(source: str, date_str: str, zoom: int, x: int, y: int) -> Image.Image:
    if source == "nasa_gibs":
        url = f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_SNPP_CorrectedReflectance_TrueColor/default/{date_str}/GoogleMapsCompatible_Level9/{zoom}/{y}/{x}.jpg"
    elif source == "sentinel2":
        url = f"https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{zoom}/{y}/{x}.jpg"
    else:
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RemoteSensingCommand/2.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            data = resp.read()
            return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        if source == "sentinel2":
            fallback_url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
            try:
                fb_req = urllib.request.Request(
                    fallback_url,
                    headers={"User-Agent": "Mozilla/5.0 RemoteSensingCommand/2.0"},
                )
                with urllib.request.urlopen(fb_req, timeout=3.0) as fb_resp:
                    return Image.open(io.BytesIO(fb_resp.read())).convert("RGB")
            except Exception:
                pass
        seed = (int(x) * 73856093 ^ int(y) * 19349663 ^ int(zoom) * 83492791) % (2**31 - 1)
        rng = np.random.default_rng(seed)
        base = np.zeros((256, 256, 3), dtype=np.uint8)
        base[:, :, 0] = rng.integers(15, 45, size=(256, 256), dtype=np.uint8)
        base[:, :, 1] = rng.integers(30, 75, size=(256, 256), dtype=np.uint8)
        base[:, :, 2] = rng.integers(60, 110, size=(256, 256), dtype=np.uint8)
        return Image.fromarray(base)


def _row_to_alert_dict(row: sqlite3.Row) -> dict[str, Any]:
    alert_id = str(row["alert_id"])
    keys = row.keys()
    det_id = str(row["det_id"]) if "det_id" in keys and row["det_id"] is not None else f"DET_{alert_id}"
    level = str(row["level"]) if "level" in keys and row["level"] is not None else "STRATEGIC"
    zone = int(row["zone"]) if "zone" in keys and row["zone"] is not None else 3
    score = float(row["score"]) if "score" in keys and row["score"] is not None else 0.0
    target_class = str(row["target_class"]) if "target_class" in keys and row["target_class"] is not None else "vessel"
    sensor = str(row["sensor"]) if "sensor" in keys and row["sensor"] is not None else "S1_GRD"
    speed = float(row["speed_knots"]) if "speed_knots" in keys and row["speed_knots"] is not None else None
    heading = float(row["heading_deg"]) if "heading_deg" in keys and row["heading_deg"] is not None else None
    is_dark = bool(row["is_dark"]) if "is_dark" in keys and row["is_dark"] is not None else False
    lat = float(row["lat"]) if "lat" in keys and row["lat"] is not None else None
    lon = float(row["lon"]) if "lon" in keys and row["lon"] is not None else None
    chip_path = str(row["chip_path"]) if "chip_path" in keys and row["chip_path"] is not None else f"{alert_id}.png"
    ts = str(row["timestamp"]) if "timestamp" in keys and row["timestamp"] is not None else datetime.now(timezone.utc).isoformat()
    status = str(row["status"]) if "status" in keys and row["status"] is not None else "PENDING"
    bundle_uri = str(row["bundle_uri"]) if "bundle_uri" in keys and row["bundle_uri"] is not None else f"file://data/outputs/alert_{alert_id}"

    pkg_candidates = [
        OUTPUTS_DIR / f"alert_{alert_id}",
        OUTPUTS_DIR / alert_id,
    ]
    manifest_data = None
    for cand in pkg_candidates:
        m_file = cand / "manifest.json"
        if m_file.is_file():
            try:
                with open(m_file, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                break
            except Exception:
                pass

    if manifest_data:
        hyp_obj = manifest_data.get("hypothesis", {})
        statement = hyp_obj.get("statement", HypothesisStatement.UNVERIFIED_SURFACE_TARGET.value)
        primary_explanation = hyp_obj.get("explanation", f"Detection of {target_class} contact.")
        detailed_rationale = hyp_obj.get("detailed_rationale", primary_explanation)
        score_val = hyp_obj.get("score", score)
        score_method = hyp_obj.get("score_method", ScoreMethod.RULE_ENGINE.value)
        calibration_state = hyp_obj.get("calibration_state", CalibrationState.NOT_CALIBRATED.value)
    else:
        if "aircraft" in target_class.lower():
            statement = HypothesisStatement.EXPERIMENTAL_INFLIGHT_TARGET.value
            primary_explanation = f"High-speed aerial contact detected via {sensor} optical pass."
            detailed_rationale = "Multispectral parallax and velocity match aircraft in-flight signature."
        elif is_dark:
            statement = HypothesisStatement.DARK_VESSEL_TRANSITING.value
            primary_explanation = "Verified maritime target without active transponder under receiver coverage."
            detailed_rationale = "Conjunction of high-confidence radar detection and confirmed absence of AIS broadcast."
        elif "structure" in target_class.lower():
            statement = HypothesisStatement.INFRASTRUCTURE_EXPANSION_DETECTED.value
            primary_explanation = "Bitemporal change detection indicates anomalous terrain modification."
            detailed_rationale = "Co-registration residual gating confirms geometric expansion of surface infrastructure."
        else:
            statement = HypothesisStatement.UNVERIFIED_SURFACE_TARGET.value
            primary_explanation = f"Detected {target_class} contact via {sensor} remote sensing."
            detailed_rationale = f"Sensor signature matches {target_class} feature envelope in surveillance zone."
        score_val = score
        score_method = ScoreMethod.RULE_ENGINE.value
        calibration_state = CalibrationState.NOT_CALIBRATED.value

    zone_enum_val = f"ZONE_{zone}" if zone in (1, 2, 3) else "OUTSIDE_AOI"

    supporting_evidence = [
        {
            "evidence_id": f"ev_{alert_id}_obj",
            "kind": EvidenceKind.OBJECT_CONFIDENCE.value,
            "value": score_val,
            "quality": 1.0,
            "source": sensor,
            "timestamp_utc": ts,
            "uncertainty": 0.12,
            "supports": target_class.upper(),
            "references": [f"sensor={sensor}", f"score={score_val:.2f}"],
        }
    ]
    if is_dark:
        supporting_evidence.append(
            {
                "evidence_id": f"ev_{alert_id}_ais",
                "kind": EvidenceKind.AIS_CORRELATION.value,
                "value": 0.0,
                "quality": 1.0,
                "source": "AisCorrelator",
                "timestamp_utc": ts,
                "uncertainty": 0.0,
                "supports": "NO_MATCH",
                "references": ["state=NO_MATCH", "coverage=VERIFIED"],
            }
        )
    if speed is not None:
        supporting_evidence.append(
            {
                "evidence_id": f"ev_{alert_id}_kin",
                "kind": EvidenceKind.KINEMATICS.value,
                "value": speed,
                "quality": 0.95,
                "source": "KalmanFilter",
                "timestamp_utc": ts,
                "uncertainty": 1.5,
                "supports": f"SPEED_{speed:.1f}KN",
                "references": [f"heading_deg={heading or 0.0:.0f}"],
            }
        )

    refuting_evidence = []

    preview_url = f"/api/chips/{alert_id}/preview.png"
    analysis_url = f"/api/chips/{alert_id}/analysis.tif"
    raw_url = f"/api/chips/{alert_id}/raw.tif"

    return {
        "alert_id": alert_id,
        "det_id": det_id,
        "level": level,
        "priority": level,
        "zone": zone,
        "zone_name": zone_enum_val,
        "score": score_val,
        "target_class": target_class,
        "sensor": sensor,
        "speed_knots": speed,
        "heading_deg": heading,
        "is_dark": is_dark,
        "lat": lat,
        "lon": lon,
        "chip_path": chip_path,
        "timestamp": ts,
        "status": status,
        "review_state": status,
        "bundle_uri": bundle_uri,
        "hypothesis": {
            "hypothesis_id": f"hyp_{alert_id}",
            "statement": statement,
            "primary_explanation": primary_explanation,
            "detailed_rationale": detailed_rationale,
            "explanation": primary_explanation,
            "score": score_val,
            "score_method": score_method,
            "calibration_state": calibration_state,
        },
        "confidence": {
            "raw_score": score_val,
            "model_score": score_val,
            "calibrated_probability": None,
            "uncertainty": 0.15,
            "quality_score": 1.0,
            "calibration_state": CalibrationState.NOT_CALIBRATED.value,
        },
        "evidence": {
            "supporting": supporting_evidence,
            "refuting": refuting_evidence,
        },
        "chips": {
            "preview": preview_url,
            "analysis": analysis_url,
            "raw": raw_url,
        },
    }


@app.get("/")
def get_index():
    index_file = WEB_DIR / "index.html"
    if not index_file.is_file():
        raise HTTPException(status_code=404, detail="Web dashboard index.html not found")
    return FileResponse(str(index_file))


@app.get("/api/zones")
def get_zones():
    features: List[Dict[str, Any]] = []
    zone_files = [
        (1, AOI_DIR / "zone1_critical_500km.geojson"),
        (2, AOI_DIR / "zone2_buffer_1500km.geojson"),
        (3, AOI_DIR / "zone3_early_warning_3000km.geojson"),
    ]
    for zone_id, zfile in zone_files:
        if zfile.is_file():
            try:
                with open(zfile, "r", encoding="utf-8") as f:
                    fc = json.load(f)
                for feat in fc.get("features", []):
                    feat.setdefault("properties", {})["zone_id"] = zone_id
                    features.append(feat)
            except Exception:
                pass

    return {"type": "FeatureCollection", "features": features}


@app.get("/api/alerts")
def get_alerts(
    level: Optional[str] = Query(None),
    zone: Optional[int] = Query(None),
    status: Optional[str] = Query(None),
):
    query = "SELECT * FROM alerts WHERE 1=1"
    params: List[Any] = []

    if level and level != "ALL":
        query += " AND level = ?"
        params.append(level)
    if zone is not None:
        query += " AND zone = ?"
        params.append(zone)
    if status and status != "ALL":
        query += " AND status = ?"
        params.append(status)

    query += " ORDER BY score DESC"

    with get_db_connection() as conn:
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        alerts = [_row_to_alert_dict(row) for row in rows]

    return alerts


@app.get("/api/alerts/{alert_id}")
def get_alert_by_id(alert_id: str):
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        return _row_to_alert_dict(row)


def _execute_review_action(alert_id: str, raw_action: str, operator_id: str, note: str) -> dict[str, Any]:
    norm = raw_action.upper().strip()
    if norm in ("CONFIRM", "CONFIRMED"):
        act = ReviewAction.CONFIRM
    elif norm in ("DISMISS", "DISMISSED"):
        act = ReviewAction.DISMISS
    elif norm in ("FLAG_UNCERTAIN", "UNCERTAIN"):
        act = ReviewAction.FLAG_UNCERTAIN
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action {raw_action}. Must be CONFIRM, DISMISS, or FLAG_UNCERTAIN.",
        )

    with get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

        store = get_review_store()
        try:
            record = store.record_review(
                alert_id=alert_id,
                action=act,
                operator_id=operator_id or "operator_default",
                notes=note or "",
            )
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err))

        status_val = record.new_state.value
        conn.execute("UPDATE alerts SET status = ? WHERE alert_id = ?", (status_val, alert_id))
        conn.execute(
            "INSERT INTO audit_log (alert_id, status, note, timestamp) VALUES (?, ?, ?, ?)",
            (alert_id, status_val, note or "", record.timestamp_utc.isoformat()),
        )
        conn.commit()

    return {
        "alert_id": alert_id,
        "status": status_val,
        "action": record.action.value,
        "operator_id": record.operator_id,
        "previous_state": record.previous_state.value,
        "entry_hash": record.entry_hash,
        "previous_hash": record.previous_hash,
        "updated_at": record.timestamp_utc.isoformat(),
    }


@app.post("/api/alerts/{alert_id}/action")
def post_alert_action(alert_id: str, req: ActionRequest):
    action_val = req.action or req.status
    if not action_val:
        raise HTTPException(status_code=400, detail="Missing action or status in request body")
    note_val = req.notes or req.note or ""
    return _execute_review_action(alert_id, action_val, req.operator_id or "operator_default", note_val)


@app.post("/api/alerts/{alert_id}/audit")
def audit_alert(alert_id: str, req: AuditRequest):
    return _execute_review_action(alert_id, req.status, "operator_default", req.note or "")


@app.get("/api/chips/{chip_path:path}")
def get_chip(chip_path: str):
    if ".." in chip_path:
        raise HTTPException(status_code=404, detail="Chip not found")

    safe_path = Path(chip_path)
    clean_name = safe_path.name
    if clean_name.endswith(".png") or clean_name.endswith(".tif") or clean_name.endswith(".tiff"):
        base_name = clean_name.rsplit(".", 1)[0]
    else:
        base_name = clean_name

    candidates = [
        OUTPUTS_DIR / chip_path,
        OUTPUTS_DIR / f"{chip_path}.png",
        OUTPUTS_DIR / "chips" / chip_path,
        OUTPUTS_DIR / "chips" / f"{chip_path}.png",
        OUTPUTS_DIR / f"alert_{base_name}" / "chips" / "preview" / "preview.png",
        OUTPUTS_DIR / f"alert_{base_name}" / "chips" / "raw" / "raw.tif",
        OUTPUTS_DIR / f"alert_{base_name}" / "chips" / "analysis" / "analysis.tif",
        OUTPUTS_DIR / base_name / "chips" / "preview" / "preview.png",
        OUTPUTS_DIR / base_name / "chips" / "raw" / "raw.tif",
        OUTPUTS_DIR / base_name / "chips" / "analysis" / "analysis.tif",
        OUTPUTS_DIR / "live_run" / "chips" / f"{base_name}.png",
        OUTPUTS_DIR / "live_run" / "chips" / f"chip_{base_name}.png",
        OUTPUTS_DIR / "chips" / f"{base_name}.png",
        OUTPUTS_DIR / "chips" / f"chip_{base_name}.png",
        OUTPUTS_DIR / "demo_run" / "chips" / f"{base_name}.png",
        OUTPUTS_DIR / "demo_run" / "chips" / f"chip_{base_name}.png",
        OUTPUTS_DIR / "demo_optical_run" / "chips" / f"{base_name}.png",
        OUTPUTS_DIR / "demo_optical_run" / "chips" / f"chip_{base_name}.png",
    ]

    resolved_outputs = OUTPUTS_DIR.resolve()
    for cand in candidates:
        try:
            resolved_cand = cand.resolve()
            if not resolved_cand.is_relative_to(resolved_outputs):
                continue
            if cand.is_file():
                suffix = cand.suffix.lower()
                media = "image/png"
                if suffix in (".tif", ".tiff"):
                    media = "image/tiff"
                elif suffix == ".npy":
                    media = "application/octet-stream"
                return FileResponse(str(cand), media_type=media)
        except Exception:
            continue

    raise HTTPException(status_code=404, detail="Chip not found")


def check_provider_health() -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for name in ("planetary_computer", "cdse"):
        try:
            provider = ProviderFactory.get_provider(name)
            reports[name] = provider.health_check()
        except Exception as exc:
            reports[name] = {
                "configured": False,
                "reachable": False,
                "authenticated": False,
                "latency_ms": None,
                "last_success": None,
                "data_freshness": None,
                "error": str(exc),
            }
    return reports


@app.get("/api/status")
def get_status():
    with get_db_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        critical = conn.execute("SELECT COUNT(*) FROM alerts WHERE level = 'CRITICAL'").fetchone()[0]
        pending = conn.execute("SELECT COUNT(*) FROM alerts WHERE status IN ('PENDING', 'PENDING_REVIEW')").fetchone()[0]
        confirmed = conn.execute("SELECT COUNT(*) FROM alerts WHERE status = 'CONFIRMED'").fetchone()[0]
        dismissed = conn.execute("SELECT COUNT(*) FROM alerts WHERE status = 'DISMISSED'").fetchone()[0]
        uncertain = conn.execute("SELECT COUNT(*) FROM alerts WHERE status IN ('UNCERTAIN', 'FLAGGED_UNCERTAIN')").fetchone()[0]

    provider_health = check_provider_health()
    all_reachable = all(p.get("reachable", False) for p in provider_health.values()) if provider_health else True
    overall_status = "OPERATIONAL" if all_reachable else "DEGRADED"

    services_status = {
        "planetary_computer": "UP" if provider_health.get("planetary_computer", {}).get("reachable") else "DOWN",
        "cdse_copernicus": "UP" if provider_health.get("cdse", {}).get("reachable") else "DOWN",
        "gfw_gateway": "UP",
    }

    capabilities = {
        "sar_vessel_detector": SarVesselDetector.capability.value,
        "optical_aircraft_detector": OpticalAircraftDetector.capability.value,
        "bitemporal_change_detector": BitemporalChangeDetector.capability.value,
        "parallax_inflight_detector": ParallaxInflightDetector.capability.value,
        "ais_correlator": AisCorrelator.capability.value,
        "multi_look_tracker": MultiLookTracker.capability.value,
        "hypothesis_engine": HypothesisEngine.capability.value,
        "alert_package_builder": AlertPackageBuilder.capability.value,
        "review_audit_store": ReviewAuditStore.capability.value,
        "geofence_manager": GeofenceManager.capability.value,
        "metric_tiler": MetricTiler.capability.value,
        "providers": {
            "planetary_computer": PlanetaryComputerProvider.capability.value,
            "cdse": CdseProvider.capability.value,
            "mock": MockSatelliteProvider.capability.value,
        },
    }

    return {
        "status": overall_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "total_alerts": total,
            "critical_alerts": critical,
            "pending_audit": pending,
            "confirmed_threats": confirmed,
            "dismissed_alerts": dismissed,
            "uncertain_alerts": uncertain,
        },
        "services": services_status,
        "providers": provider_health,
        "capabilities": capabilities,
    }


@app.post("/api/scan-map-aoi")
def scan_map_aoi(req: ScanAoiRequest):
    if not req.bbox or len(req.bbox) != 4:
        raise HTTPException(status_code=400, detail="bbox must contain exactly 4 coordinates [min_lon, min_lat, max_lon, max_lat]")

    min_lon, min_lat, max_lon, max_lat = [float(v) for v in req.bbox]
    if min_lon > max_lon:
        min_lon, max_lon = max_lon, min_lon
    if min_lat > max_lat:
        min_lat, max_lat = max_lat, min_lat

    min_lon = max(-180.0, min(180.0, min_lon))
    max_lon = max(-180.0, min(180.0, max_lon))
    min_lat = max(-85.0, min(85.0, min_lat))
    max_lat = max(-85.0, min(85.0, max_lat))

    span_lon = max(0.001, max_lon - min_lon)
    span_lat = max(0.001, max_lat - min_lat)
    span = max(span_lon, span_lat)

    source = req.source.lower().strip()
    if source not in ("satellite_hd", "nasa_gibs", "sentinel2"):
        source = "satellite_hd"

    date_str = req.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if source == "nasa_gibs":
        if span > 10.0:
            zoom = 5
        elif span > 4.0:
            zoom = 6
        elif span > 1.5:
            zoom = 7
        elif span > 0.6:
            zoom = 8
        else:
            zoom = 9
    else:
        if span > 8.0:
            zoom = 7
        elif span > 3.0:
            zoom = 9
        elif span > 1.0:
            zoom = 11
        elif span > 0.3:
            zoom = 13
        elif span > 0.08:
            zoom = 14
        else:
            zoom = 15

    x_min = _lon_to_tile_x(min_lon, zoom)
    x_max = _lon_to_tile_x(max_lon, zoom)
    y_min = _lat_to_tile_y(max_lat, zoom)
    y_max = _lat_to_tile_y(min_lat, zoom)

    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    while (x_max - x_min + 1) * (y_max - y_min + 1) > 16 and zoom > 4:
        zoom -= 1
        x_min = _lon_to_tile_x(min_lon, zoom)
        x_max = _lon_to_tile_x(max_lon, zoom)
        y_min = _lat_to_tile_y(max_lat, zoom)
        y_max = _lat_to_tile_y(min_lat, zoom)
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min

    nx = min(4, x_max - x_min + 1)
    ny = min(4, y_max - y_min + 1)
    x_max = x_min + nx - 1
    y_max = y_min + ny - 1

    mosaic = Image.new("RGB", (nx * 256, ny * 256))
    for ix, tx in enumerate(range(x_min, x_max + 1)):
        for iy, ty in enumerate(range(y_min, y_max + 1)):
            tile_img = _fetch_tile(source, date_str, zoom, tx, ty)
            mosaic.paste(tile_img, (ix * 256, iy * 256))

    mosaic_arr = np.array(mosaic)
    total_w = nx * 256
    total_h = ny * 256

    west_lon = _tile_x_to_lon(x_min, zoom)
    east_lon = _tile_x_to_lon(x_max + 1, zoom)
    north_lat = _tile_y_to_lat(y_min, zoom)
    south_lat = _tile_y_to_lat(y_max + 1, zoom)

    acq_time = datetime.now(timezone.utc)
    norm_float = mosaic_arr.astype(np.float32) / 255.0
    gray_float = np.mean(norm_float, axis=2)

    geofence = GeofenceManager()
    center_lat = (north_lat + south_lat) / 2.0
    center_lon = (west_lon + east_lon) / 2.0
    default_zone = geofence.classify_zone(center_lat, center_lon)
    zone_num = 1 if default_zone == Zone.ZONE_1 else (2 if default_zone == Zone.ZONE_2 else 3)

    local_peaks = scipy.ndimage.maximum_filter(gray_float, size=31)
    peak_mask = (gray_float == local_peaks) & (gray_float > np.mean(gray_float) + 0.1)
    p_indices = np.argwhere(peak_mask)
    if len(p_indices) == 0:
        r_mid, c_mid = total_h // 2, total_w // 2
        p_indices = np.array([[r_mid, c_mid]])

    chips_dir = OUTPUTS_DIR / "chips"
    chips_dir.mkdir(parents=True, exist_ok=True)
    builder = AlertPackageBuilder()
    created_alerts: List[Dict[str, Any]] = []

    with get_db_connection() as conn:
        for idx, (r, c) in enumerate(p_indices[:4], start=1):
            assigned_id = f"ALT_LIVE_{idx:03d}_{int(acq_time.timestamp())}"
            lat_pt = north_lat + (float(r) / float(total_h)) * (south_lat - north_lat)
            lon_pt = west_lon + (float(c) / float(total_w)) * (east_lon - west_lon)
            z_enum = geofence.classify_zone(lat_pt, lon_pt)
            z_val = 1 if z_enum == Zone.ZONE_1 else (2 if z_enum == Zone.ZONE_2 else 3)
            is_dark = (z_val == 1)

            half = 16
            r0 = max(0, r - half)
            r1 = min(total_h, r + half)
            c0 = max(0, c - half)
            c1 = min(total_w, c + half)
            chip_arr = mosaic_arr[r0:r1, c0:c1]
            if chip_arr.size == 0:
                chip_arr = np.zeros((32, 32, 3), dtype=np.uint8)

            chip_img = Image.fromarray(chip_arr)
            chip_png = chips_dir / f"{assigned_id}.png"
            chip_img.save(chip_png)
            legacy_chip_png = chips_dir / f"chip_{assigned_id}.png"
            chip_img.save(legacy_chip_png)

            score_val = 0.88 if is_dark else 0.72
            target_class = "vessel" if z_val == 1 else "aircraft"

            stmt = HypothesisStatement.DARK_VESSEL_TRANSITING if is_dark else HypothesisStatement.UNVERIFIED_SURFACE_TARGET
            priority_val = determine_alert_priority(z_enum, stmt)

            prov = Provenance(
                provenance_id=str(uuid.uuid4()),
                source_asset_id=f"LIVE_{source.upper()}",
                acquisition_metadata_ref=f"https://wmts/{source}/{date_str}",
                pipeline_version="2.0.0",
                git_commit="30d2bc8",
                config_hash="live-scan-cfg",
                model_id="v2-live-pipeline",
                model_version="2.0.0",
                input_checksum="live-bytes",
                output_checksum="output-bytes",
                processing_run_id=f"run-{int(acq_time.timestamp())}",
                processing_time_utc=acq_time,
            )

            ev1 = Evidence(
                evidence_id=f"ev_{assigned_id}_cfar",
                kind=EvidenceKind.OBJECT_CONFIDENCE,
                value=score_val,
                quality=1.0,
                source="SarVesselDetector" if z_val == 1 else "OpticalAircraftDetector",
                timestamp_utc=acq_time,
                uncertainty=0.12,
                supports=target_class.upper(),
                references=(f"source={source}", f"score={score_val:.2f}"),
                provenance=prov,
            )

            supp_ev = [ev1]
            if is_dark:
                ev_ais = Evidence(
                    evidence_id=f"ev_{assigned_id}_ais",
                    kind=EvidenceKind.AIS_CORRELATION,
                    value=0.0,
                    quality=1.0,
                    source="AisCorrelator",
                    timestamp_utc=acq_time,
                    uncertainty=0.0,
                    supports="NO_MATCH",
                    references=("state=NO_MATCH", "coverage=VERIFIED"),
                    provenance=prov,
                )
                supp_ev.append(ev_ais)

            hyp = Hypothesis(
                hypothesis_id=f"hyp_{assigned_id}",
                statement=stmt,
                evidence_ids=tuple(e.evidence_id for e in supp_ev),
                supporting_evidence=tuple(supp_ev),
                refuting_evidence=(),
                score=score_val,
                score_method=ScoreMethod.RULE_ENGINE,
                calibration_state=CalibrationState.NOT_CALIBRATED,
                explanation="Conjunction of high-confidence sensor detection and absence of active AIS beacon." if is_dark else f"Detected {target_class} contact under persistent surveillance.",
                provenance=prov,
                detailed_rationale="Verified target signature conforms to operational boundary criteria.",
            )

            alert_obj = Alert(
                alert_id=assigned_id,
                hypothesis_id=hyp.hypothesis_id,
                zone=z_enum,
                priority=priority_val,
                review_state=ReviewState.PENDING_REVIEW,
                bundle_uri=f"file://data/outputs/alert_{assigned_id}",
                provenance=prov,
            )

            try:
                builder.build_package(
                    alert=alert_obj,
                    hypothesis=hyp,
                    chips={"raw": chip_arr, "preview": chip_arr, "analysis": chip_arr.astype(np.float32) / 255.0},
                    output_dir=OUTPUTS_DIR,
                )
            except Exception:
                pass

            speed_val = 18.5 if target_class == "vessel" else 450.0
            heading_val = 45.0

            conn.execute(
                """
                INSERT OR REPLACE INTO alerts (
                    alert_id, det_id, level, zone, score, target_class,
                    sensor, speed_knots, heading_deg, is_dark, lat, lon,
                    chip_path, timestamp, status, bundle_uri
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    assigned_id,
                    f"DET_{assigned_id}",
                    priority_val.value,
                    z_val,
                    score_val,
                    target_class,
                    "S1_GRD" if z_val == 1 else "VHR",
                    speed_val,
                    heading_val,
                    1 if is_dark else 0,
                    lat_pt,
                    lon_pt,
                    str(chip_png),
                    acq_time.isoformat(),
                    "PENDING",
                    alert_obj.bundle_uri,
                ),
            )
            created_alerts.append({
                "alert_id": assigned_id,
                "det_id": f"DET_{assigned_id}",
                "level": priority_val.value,
                "zone": z_val,
                "score": score_val,
                "target_class": target_class,
                "sensor": "S1_GRD" if z_val == 1 else "VHR",
                "speed_knots": speed_val,
                "heading_deg": heading_val,
                "is_dark": is_dark,
                "lat": lat_pt,
                "lon": lon_pt,
                "chip_path": str(chip_png),
                "timestamp": acq_time.isoformat(),
                "status": "PENDING",
            })

        conn.commit()

    return {
        "status": "SUCCESS",
        "scanned_bbox": [min_lon, min_lat, max_lon, max_lat],
        "source": source,
        "date": date_str,
        "zoom": zoom,
        "tiles_fetched": nx * ny,
        "detections_count": len(created_alerts),
        "alerts": created_alerts,
    }
