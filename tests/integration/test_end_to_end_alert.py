import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
from shapely.geometry import shape

from src.detectors.sar_vessel import SarVesselDetector
from src.detectors.parallax_detector import ParallaxInflightDetector
from src.engine.tracker import TemporalTracker
from src.engine.fusion import AisFusionEngine, AisRecord
from src.core.geofence import GeofenceManager
from src.engine.anomaly_scorer import AnomalyScorer
from src.engine.alert_generator import AlertGenerator
from tests.mocks.mock_data import generate_synthetic_sar_scene, generate_synthetic_parallax_scene


def test_end_to_end_sar_vessel_pipeline(tmp_path: Path):
    t0 = datetime(2026, 10, 22, 4, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(seconds=1800)

    sar_t0 = generate_synthetic_sar_scene(
        height=120,
        width=120,
        targets=[(40, 40, 6, 3, 500.0)],
        seed=10,
    )
    sar_t1 = generate_synthetic_sar_scene(
        height=120,
        width=120,
        targets=[(45, 45, 6, 3, 500.0)],
        seed=20,
    )

    detector = SarVesselDetector()
    t_mat = (0.0001, 0.0, 108.0, 0.0, 0.0001, 16.0)
    dets_t0 = detector.detect(sar_t0, sensor="S1_GRD", source_tile="scene_t0", acquisition_time=t0, transform_matrix=t_mat)
    dets_t1 = detector.detect(sar_t1, sensor="S1_GRD", source_tile="scene_t1", acquisition_time=t1, transform_matrix=t_mat)

    assert len(dets_t0) >= 1
    assert len(dets_t1) >= 1

    tracker = TemporalTracker(max_distance_meters=50000.0, max_dt_seconds=3600.0)
    tracks = tracker.associate(dets_t0, dets_t1)
    assert len(tracks) >= 1
    active_track = tracks[0]
    assert active_track.velocity_mps is not None
    assert active_track.heading_deg is not None

    ais_records = [
        AisRecord(
            mmsi="123456789",
            lat=16.0,
            lon=108.0,
            timestamp=t1,
            vessel_name="TEST_VESSEL",
        )
    ]
    fusion = AisFusionEngine(is_nrt_feed_active=True)
    correlated = fusion.correlate_batch(dets_t1, ais_records)
    assert len(correlated) == len(dets_t1)

    geofence = GeofenceManager()
    scorer = AnomalyScorer(
        weights={"w_z": 0.40, "w_d": 0.25, "w_v": 0.15, "w_c": 0.10, "w_b": 0.10},
        reference_distance_km=500.0,
    )

    chips_dir = tmp_path / "chips"
    alert_gen = AlertGenerator(output_dir=chips_dir)
    alerts = []

    for corr in correlated:
        det = corr.detection
        pt = det.geom_wgs84.centroid
        lat, lon = pt.y, pt.x
        zone_raw = geofence.classify_zone(lat, lon)
        zone = 1 if zone_raw in (0, 1) else zone_raw
        dist_m = geofence.compute_geodesic_distance_m(lat, lon)
        dist_km = dist_m / 1000.0

        v_score = min(1.0, active_track.velocity_mps / 15.0) if active_track.velocity_mps else 0.0
        score_s = scorer.calculate_score(
            zone=zone,
            is_dark=corr.is_dark,
            velocity_score=v_score,
            geodesic_distance_km=dist_km,
        )

        alert = alert_gen.generate_alert(
            detection=det,
            zone=zone,
            score_s=score_s,
            source_image=sar_t1,
            is_dark=corr.is_dark,
        )
        alerts.append(alert)

    assert len(alerts) >= 1
    sample_alert = alerts[0]
    assert sample_alert.zone in (1, 2, 3)
    assert sample_alert.level in ("CRITICAL", "HIGH", "STRATEGIC")
    assert 0.0 <= sample_alert.score_s <= 100.0
    assert Path(sample_alert.chip_path).exists()
    assert Path(sample_alert.chip_path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    geom_shapely = shape(sample_alert.geojson_feature["geometry"])
    assert geom_shapely.is_valid

    reviewed_alert = alert_gen.update_audit_status(sample_alert, "CONFIRMED")
    assert reviewed_alert.audit_status == "CONFIRMED"


def test_end_to_end_optical_parallax_pipeline(tmp_path: Path):
    scene = generate_synthetic_parallax_scene(height=100, width=100)
    detector = ParallaxInflightDetector()
    detections = detector.detect(scene)

    assert len(detections) >= 1
    det = detections[0]

    geofence = GeofenceManager()
    centroid = det.geom_wgs84.centroid
    zone_raw = geofence.classify_zone(centroid.y, centroid.x)
    zone = 1 if zone_raw in (0, 1) else zone_raw
    dist_m = geofence.compute_geodesic_distance_m(centroid.y, centroid.x)

    scorer = AnomalyScorer(
        weights={"w_z": 0.40, "w_d": 0.25, "w_v": 0.15, "w_c": 0.10, "w_b": 0.10},
        reference_distance_km=500.0,
    )
    score_s = scorer.calculate_score(zone=zone, is_dark=None, geodesic_distance_km=dist_m / 1000.0)

    alert_gen = AlertGenerator(output_dir=tmp_path / "chips")
    alert = alert_gen.generate_alert(
        detection=det,
        zone=zone,
        score_s=score_s,
        source_image=scene,
        is_dark=None,
    )

    assert alert.level in ("CRITICAL", "HIGH", "STRATEGIC")
    assert Path(alert.chip_path).exists()
    assert alert.geojson_feature["properties"]["target_class"] == "aircraft_inflight"
