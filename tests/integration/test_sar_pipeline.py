from datetime import datetime, timezone
from src.core.cfar import OSCFARDetector
from src.core.contracts import Alert
from src.core.geofence import GeofenceManager
from src.detectors.sar_vessel import SarVesselDetector
from src.engine.anomaly_scorer import AnomalyScorer
from src.engine.fusion import AisFusionEngine, AisRecord
from tests.mocks.mock_data import generate_synthetic_sar_scene


def test_end_to_end_sar_pipeline():
    scene = generate_synthetic_sar_scene(
        height=70,
        width=70,
        targets=[
            (20, 20, 4, 2, 500.0),
            (50, 50, 2, 5, 550.0),
        ],
    )

    cfar = OSCFARDetector(guard_m=15.0, train_m=45.0, pixel_spacing_m=10.0, pfa=1e-4)
    detector = SarVesselDetector(
        weights_config_path="configs/model_registry.yaml",
        cfar_detector=cfar,
        pixel_spacing_m=10.0,
    )

    transform = (0.001, 0.0, 108.0, 0.0, -0.001, 16.0)
    acq_time = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)

    detections = detector.detect(
        scene,
        sensor="S1_GRD",
        source_tile="SYNTH_S1_001",
        acquisition_time=acq_time,
        transform_matrix=transform,
    )

    assert len(detections) >= 2

    ais_records = [
        AisRecord(
            mmsi="412000001",
            lat=16.0 - (21.5 * 0.001),
            lon=108.0 + (21.0 * 0.001),
            timestamp=acq_time,
            vessel_name="AUTHORIZED_CARGO",
        )
    ]

    fusion = AisFusionEngine(
        distance_threshold_m=1500.0,
        time_threshold_seconds=1800.0,
        is_nrt_feed_active=True,
    )

    correlated_list = fusion.correlate_batch(detections, ais_records)
    assert len(correlated_list) == len(detections)

    matched = [c for c in correlated_list if c.classification == "AIS_MATCHED"]
    dark = [c for c in correlated_list if c.classification == "DARK_VESSEL"]
    assert len(matched) >= 1
    assert len(dark) >= 1

    geofence = GeofenceManager(center_lat=16.0, center_lon=108.0)
    scorer = AnomalyScorer(
        weights={
            "w_z": 0.40,
            "w_d": 0.25,
            "w_v": 0.15,
            "w_c": 0.10,
            "w_b": 0.10,
        },
        reference_distance_km=500.0,
    )

    alerts = []
    for idx, item in enumerate(dark):
        c_lon, c_lat = item.detection.geom_wgs84.centroid.x, item.detection.geom_wgs84.centroid.y
        zone = geofence.classify_zone(c_lat, c_lon)
        dist_km = geofence.compute_geodesic_distance_m(c_lat, c_lon) / 1000.0

        score = scorer.calculate_score(
            zone=zone if zone > 0 else 1,
            is_dark=item.is_dark,
            velocity_score=0.2,
            change_score=0.0,
            geodesic_distance_km=dist_km,
        )

        level = "CRITICAL" if score >= 70.0 else ("HIGH" if score >= 40.0 else "STRATEGIC")

        alert = Alert(
            alert_id=f"ALT_{idx:04d}",
            zone=zone if zone in (1, 2, 3) else 1,
            level=level,
            score_s=score,
            is_dark=item.is_dark,
            geojson_feature={
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [c_lon, c_lat],
                },
                "properties": {
                    "det_id": item.detection.det_id,
                    "classification": item.classification,
                },
            },
            chip_path="/data/chips/mock_chip.png",
            audit_status="PENDING",
        )
        alerts.append(alert)

    assert len(alerts) >= 1
    for alert in alerts:
        assert alert.zone in (1, 2, 3)
        assert alert.score_s > 0.0
        assert alert.is_dark is True
        assert alert.audit_status == "PENDING"
