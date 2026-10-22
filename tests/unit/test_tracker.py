from datetime import datetime, timezone, timedelta
from shapely.geometry import Point
from src.core.contracts import Detection
from src.engine.tracker import TemporalTracker


def _create_detection(
    det_id: str,
    lon: float,
    lat: float,
    acq_time: datetime,
    target_class: str = "vessel",
    heading_deg: float = 45.0,
) -> Detection:
    return Detection(
        det_id=det_id,
        sensor="S1_GRD",
        geom_px=(10.0, 10.0, 20.0, 20.0),
        geom_wgs84=Point(lon, lat),
        crs="EPSG:4326",
        confidence=0.9,
        length_m=50.0,
        heading_deg=heading_deg,
        target_class=target_class,
        source_tile="tile_01",
        acquisition_time=acq_time,
    )


def test_tracker_association_success():
    t0 = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(seconds=1800)

    det1 = _create_detection("D1", 108.0, 16.0, t0)
    det2 = _create_detection("D2", 108.1, 16.0, t1)

    tracker = TemporalTracker(max_distance_meters=50000.0, max_dt_seconds=3600.0)
    tracks = tracker.associate([det1], [det2])

    assert len(tracks) == 1
    tr = tracks[0]
    assert tr.track_id == "TRK-D1-D2"
    assert len(tr.detections) == 2
    assert tr.time_delta_seconds == 1800.0
    assert tr.velocity_mps is not None
    assert tr.velocity_mps > 0.0
    assert tr.heading_deg is not None
    assert 80.0 <= tr.heading_deg <= 100.0


def test_tracker_exceeds_max_dt():
    t0 = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(seconds=7500)

    det1 = _create_detection("D1", 108.0, 16.0, t0)
    det2 = _create_detection("D2", 108.05, 16.0, t1)

    tracker = TemporalTracker(max_distance_meters=50000.0, max_dt_seconds=7200.0)
    tracks = tracker.associate([det1], [det2])

    assert len(tracks) == 2
    for tr in tracks:
        assert len(tr.detections) == 1
        assert tr.velocity_mps is None
        assert tr.time_delta_seconds == 0.0


def test_tracker_exceeds_max_distance():
    t0 = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(seconds=1800)

    det1 = _create_detection("D1", 108.0, 16.0, t0)
    det2 = _create_detection("D2", 112.0, 16.0, t1)

    tracker = TemporalTracker(max_distance_meters=20000.0, max_dt_seconds=3600.0)
    tracks = tracker.associate([det1], [det2])

    assert len(tracks) == 2
    assert all(tr.velocity_mps is None for tr in tracks)


def test_tracker_reverse_time_input():
    t0 = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(seconds=1200)

    det1 = _create_detection("D1", 108.0, 16.0, t0)
    det2 = _create_detection("D2", 108.02, 16.0, t1)

    tracker = TemporalTracker(max_distance_meters=50000.0, max_dt_seconds=3600.0)
    tracks = tracker.associate([det2], [det1])

    assert len(tracks) == 1
    assert tracks[0].track_id == "TRK-D1-D2"
    assert tracks[0].time_delta_seconds == 1200.0


def test_tracker_track_method_empty_and_single_and_sequence():
    tracker = TemporalTracker()
    assert tracker.track([]) == []

    t0 = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)
    d1 = _create_detection("D1", 108.0, 16.0, t0)
    d2 = _create_detection("D2", 108.5, 16.5, t0)
    single_time_tracks = tracker.track([d1, d2])
    assert len(single_time_tracks) == 2
    assert all(t.velocity_mps is None for t in single_time_tracks)

    t1 = t0 + timedelta(seconds=600)
    d3 = _create_detection("D3", 108.01, 16.0, t1)
    seq_tracks = tracker.track([d1, d3])
    assert any(t.velocity_mps is not None for t in seq_tracks)
