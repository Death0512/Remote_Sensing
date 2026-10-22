from datetime import datetime, timezone
from src.core.geofence import GeofenceManager
from src.detectors.optical_aircraft import OpticalAircraftDetector
from src.detectors.parallax_detector import ParallaxInflightDetector
from tests.mocks.mock_data import (
    generate_synthetic_optical_airport,
    generate_synthetic_parallax_scene,
)


def test_optical_pipeline_parked_aircraft_geofencing():
    scene = generate_synthetic_optical_airport(
        height=100,
        width=100,
        aircraft_positions=[(40, 40, 15, 8)],
    )

    detector = OpticalAircraftDetector(
        weights_config_path="configs/model_registry.yaml",
        pixel_spacing_m=0.5,
        intensity_threshold=0.6,
        min_area_px=15,
    )

    transform = (0.0001, 0.0, 108.0, 0.0, -0.0001, 16.0)
    acq_time = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)

    detections = detector.detect(
        scene,
        sensor="VHR",
        source_tile="VHR_AIRBASE_01",
        acquisition_time=acq_time,
        transform_matrix=transform,
    )

    assert len(detections) >= 1
    det = detections[0]

    geofence = GeofenceManager(center_lat=16.0, center_lon=108.0)
    centroid = det.geom_wgs84.centroid
    zone = geofence.classify_zone(centroid.y, centroid.x)

    assert zone in (1, 2, 3)
    assert zone == 1


def test_optical_pipeline_inflight_parallax_geofencing():
    scene = generate_synthetic_parallax_scene(
        height=100,
        width=100,
        triplets=[((30, 30), (33, 34), (36, 38))],
    )

    detector = ParallaxInflightDetector(
        pixel_spacing_m=10.0,
        inter_band_delay_seconds=0.52,
        min_velocity_kmh=300.0,
        max_velocity_kmh=1000.0,
    )

    transform = (0.001, 0.0, 109.0, 0.0, -0.001, 16.5)
    acq_time = datetime(2026, 10, 22, 11, 0, 0, tzinfo=timezone.utc)

    detections = detector.detect(
        scene,
        sensor="S2_L1C",
        source_tile="S2_CRITICAL_CORRIDOR",
        acquisition_time=acq_time,
        transform_matrix=transform,
        peak_threshold=0.8,
    )

    assert len(detections) >= 1
    det = detections[0]
    assert det.target_class == "aircraft_inflight"

    geofence = GeofenceManager(center_lat=16.0, center_lon=108.0)
    centroid = det.geom_wgs84.centroid
    zone = geofence.classify_zone(centroid.y, centroid.x)

    assert zone in (1, 2)
