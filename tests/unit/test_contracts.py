from datetime import datetime, timezone
from shapely.geometry import Point
from src.core.contracts import Detection, Track, Alert


def test_detection_dataclass_instantiation():
    geom = Point(108.0, 16.0)
    det = Detection(
        det_id="det_001",
        sensor="S1_GRD",
        geom_px=(10.0, 10.0, 50.0, 50.0),
        geom_wgs84=geom,
        crs="EPSG:4326",
        confidence=0.95,
        length_m=120.0,
        heading_deg=45.0,
        target_class="vessel",
        source_tile="tile_01.tif",
        acquisition_time=datetime.now(timezone.utc),
    )
    assert det.det_id == "det_001"
    assert det.confidence == 0.95
    assert det.target_class == "vessel"

    det_new_structure = Detection(
        det_id="det_002",
        sensor="S1_RTC",
        geom_px=(10.0, 10.0, 50.0, 50.0),
        geom_wgs84=geom,
        crs="EPSG:4326",
        confidence=0.90,
        length_m=50.0,
        heading_deg=None,
        target_class="new_structure",
        source_tile="tile_02.tif",
        acquisition_time=datetime.now(timezone.utc),
    )
    assert det_new_structure.target_class == "new_structure"


def test_track_dataclass_instantiation():
    geom = Point(108.0, 16.0)
    det = Detection(
        det_id="det_001",
        sensor="S1_GRD",
        geom_px=(10.0, 10.0, 50.0, 50.0),
        geom_wgs84=geom,
        crs="EPSG:4326",
        confidence=0.95,
        length_m=120.0,
        heading_deg=45.0,
        target_class="vessel",
        source_tile="tile_01.tif",
        acquisition_time=datetime.now(timezone.utc),
    )
    track = Track(
        track_id="trk_001",
        detections=[det],
        velocity_mps=12.5,
        heading_deg=45.0,
        time_delta_seconds=600.0,
    )
    assert track.track_id == "trk_001"
    assert len(track.detections) == 1
    assert track.velocity_mps == 12.5


def test_alert_dataclass_instantiation():
    alert = Alert(
        alert_id="alt_001",
        zone=1,
        level="CRITICAL",
        score_s=88.5,
        is_dark=True,
        geojson_feature={"type": "Feature", "geometry": {"type": "Point", "coordinates": [108.0, 16.0]}},
        chip_path="data/outputs/chips/alt_001.png",
        audit_status="PENDING",
    )
    assert alert.alert_id == "alt_001"
    assert alert.zone == 1
    assert alert.level == "CRITICAL"
