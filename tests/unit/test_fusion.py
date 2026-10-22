from datetime import datetime, timezone
from shapely.geometry import Polygon
from src.core.contracts import Detection
from src.engine.fusion import AisFusionEngine, AisRecord


def _make_detection(det_id: str, lon: float, lat: float, acq_time: datetime) -> Detection:
    delta = 0.001
    poly = Polygon(
        [
            (lon - delta, lat - delta),
            (lon + delta, lat - delta),
            (lon + delta, lat + delta),
            (lon - delta, lat + delta),
            (lon - delta, lat - delta),
        ]
    )
    return Detection(
        det_id=det_id,
        sensor="S1_GRD",
        geom_px=(10.0, 10.0, 20.0, 20.0),
        geom_wgs84=poly,
        crs="EPSG:4326",
        confidence=0.9,
        length_m=50.0,
        heading_deg=90.0,
        target_class="vessel",
        source_tile="TILE_01",
        acquisition_time=acq_time,
    )


def test_fusion_correlated_match():
    t0 = datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc)
    det = _make_detection("DET_01", lon=108.0, lat=16.0, acq_time=t0)

    ais_record = AisRecord(
        mmsi="123456789",
        lat=16.001,
        lon=108.001,
        timestamp=t0,
    )

    engine = AisFusionEngine(distance_threshold_m=1000.0, is_nrt_feed_active=True)
    res = engine.correlate(det, [ais_record])

    assert res.classification == "AIS_MATCHED"
    assert res.is_dark is False
    assert res.matched_ais is not None
    assert res.matched_ais.mmsi == "123456789"
    assert res.distance_m is not None
    assert res.distance_m < 1000.0


def test_fusion_dark_vessel_nrt_active():
    t0 = datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc)
    det = _make_detection("DET_DARK", lon=108.0, lat=16.0, acq_time=t0)

    ais_far = AisRecord(
        mmsi="999999999",
        lat=17.0,
        lon=109.0,
        timestamp=t0,
    )

    engine = AisFusionEngine(distance_threshold_m=1000.0, is_nrt_feed_active=True)
    res = engine.correlate(det, [ais_far])

    assert res.classification == "DARK_VESSEL"
    assert res.is_dark is True
    assert res.matched_ais is None


def test_fusion_uncorrelated_candidate_nrt_inactive():
    t0 = datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc)
    det = _make_detection("DET_UNCORRELATED", lon=108.0, lat=16.0, acq_time=t0)

    engine = AisFusionEngine(distance_threshold_m=1000.0, is_nrt_feed_active=False)
    res = engine.correlate(det, [])

    assert res.classification == "AIS_UNCORRELATED_CANDIDATE"
    assert res.is_dark is None
    assert res.matched_ais is None


def test_fusion_naive_and_aware_datetime():
    t_aware = datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc)
    t_naive = datetime(2026, 10, 22, 12, 0, 0)
    det = _make_detection("DET_NAIVE", lon=108.0, lat=16.0, acq_time=t_naive)
    ais_record = AisRecord(
        mmsi="123456789",
        lat=16.001,
        lon=108.001,
        timestamp=t_aware,
    )
    engine = AisFusionEngine(distance_threshold_m=1000.0)
    res = engine.correlate(det, [ais_record])
    assert res.classification == "AIS_MATCHED"

