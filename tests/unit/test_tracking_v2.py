from datetime import datetime, timedelta, timezone
import numpy as np
import pytest
from shapely.geometry import Point

from src.domain.confidence import ConfidenceVector
from src.domain.detection import Detection
from src.domain.enums import Capability, SensorType, TargetClass, TrackLifecycle
from src.domain.provenance import Provenance
from src.tracking.association.hungarian import GatedHungarianAssociator
from src.tracking.filters.kalman import KalmanFilter2D
from src.tracking.tracker import MultiLookTracker


def _create_provenance(prov_id: str = "prov-001") -> Provenance:
    return Provenance(
        provenance_id=prov_id,
        source_asset_id="asset-test",
        acquisition_metadata_ref="s3://test/meta.xml",
        pipeline_version="2.0.0",
        git_commit="fedcba9",
        config_hash="cfg-trk-01",
        model_id=None,
        model_version=None,
        input_checksum="input-chk",
        output_checksum="output-chk",
        processing_run_id="run-trk-test",
        processing_time_utc=datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc),
    )


def _create_detection(
    det_id: str,
    lon: float,
    lat: float,
    acq_time: datetime,
) -> Detection:
    return Detection(
        detection_id=det_id,
        candidate_id=f"cand-{det_id}",
        sensor=SensorType.S1_SAR,
        geom_wgs84=Point(lon, lat),
        geom_px=(100.0, 100.0, 120.0, 110.0),
        crs="EPSG:4326",
        target_class=TargetClass.VESSEL,
        confidence=ConfidenceVector(raw_score=0.9),
        attributes=(("length_m", 50.0),),
        acquisition_time_utc=acq_time,
        provenance=_create_provenance(f"prov-{det_id}"),
    )


def test_kalman_filter_predict_update_convergence() -> None:
    dt = 1.0
    true_vx = 10.0
    true_vy = 5.0
    true_x = 0.0
    true_y = 0.0

    kf = KalmanFilter2D(
        initial_state=np.array([0.0, 0.0, 0.0, 0.0]),
        initial_covariance=np.diag([100.0, 100.0, 100.0, 100.0]),
        process_noise_std=0.1,
        measurement_noise_std=2.0,
    )

    rng = np.random.default_rng(42)
    for _ in range(50):
        true_x += true_vx * dt
        true_y += true_vy * dt

        kf.predict(dt)

        noise_x = rng.normal(0, 2.0)
        noise_y = rng.normal(0, 2.0)
        meas = np.array([true_x + noise_x, true_y + noise_y])
        kf.update(meas)

    est_state = kf.state.flatten()
    assert abs(est_state[0] - true_x) < 5.0
    assert abs(est_state[1] - true_y) < 5.0
    assert abs(est_state[2] - true_vx) < 1.0
    assert abs(est_state[3] - true_vy) < 1.0


def test_spatial_gating_rejects_distant_outliers() -> None:
    associator = GatedHungarianAssociator()

    class MockTrack:
        def __init__(self, x: float, y: float):
            self.predicted_pos = (x, y)

    class MockDetection:
        def __init__(self, x: float, y: float):
            self.pos = (x, y)

    tracks = [MockTrack(0.0, 0.0), MockTrack(1000.0, 1000.0)]
    detections = [
        MockDetection(50.0, 50.0),
        MockDetection(50000.0, 50000.0),
    ]

    matched, unmatched_trks, unmatched_dets = associator.associate(
        tracks, detections, gate_distance_m=500.0
    )

    assert matched == [(0, 0)]
    assert 1 in unmatched_trks
    assert 1 in unmatched_dets


def test_track_lifecycle_transitions() -> None:
    tracker = MultiLookTracker(
        gate_distance_m=1000.0,
        max_dt_seconds=3600.0,
        max_coast_misses=2,
        center_lat=16.0,
        center_lon=108.0,
    )

    t0 = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)
    d0 = _create_detection("d0", 108.0, 16.0, t0)

    tracks_0 = tracker.update([d0], t0)
    assert len(tracks_0) == 1
    assert tracks_0[0].lifecycle == TrackLifecycle.TENTATIVE
    assert tracker.capability == Capability.IMPLEMENTED

    t1 = t0 + timedelta(minutes=10)
    d1 = _create_detection("d1", 108.001, 16.001, t1)

    tracks_1 = tracker.update([d1], t1)
    assert len(tracks_1) == 1
    assert tracks_1[0].lifecycle == TrackLifecycle.CONFIRMED
    assert len(tracks_1[0].detection_ids) == 2

    t2 = t1 + timedelta(minutes=10)
    tracks_2 = tracker.update([], t2)
    assert len(tracks_2) == 1
    assert tracks_2[0].lifecycle == TrackLifecycle.COASTING

    t3 = t2 + timedelta(minutes=10)
    tracks_3 = tracker.update([], t3)
    assert len(tracks_3) == 1
    assert tracks_3[0].lifecycle == TrackLifecycle.COASTING

    t4 = t3 + timedelta(minutes=10)
    tracks_4 = tracker.update([], t4)
    assert len(tracks_4) == 1
    assert tracks_4[0].lifecycle == TrackLifecycle.TERMINATED

    t5 = t4 + timedelta(minutes=10)
    tracks_5 = tracker.update([], t5)
    assert len(tracks_5) == 0


def test_unbounded_dt_prevents_fabricated_velocity() -> None:
    tracker = MultiLookTracker(
        gate_distance_m=50000.0,
        max_dt_seconds=3600.0,
        max_coast_misses=2,
        center_lat=16.0,
        center_lon=108.0,
    )

    t0 = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)
    d0 = _create_detection("d0", 108.0, 16.0, t0)
    tracks_0 = tracker.update([d0], t0)
    assert len(tracks_0) == 1
    assert tracks_0[0].lifecycle == TrackLifecycle.TENTATIVE

    t1 = t0 + timedelta(hours=3)
    d1 = _create_detection("d1", 108.01, 16.01, t1)
    tracks_1 = tracker.update([d1], t1)

    assert len(tracks_1) == 2
    terminated_track = [t for t in tracks_1 if t.lifecycle == TrackLifecycle.TERMINATED][0]
    tentative_track = [t for t in tracks_1 if t.lifecycle == TrackLifecycle.TENTATIVE][0]

    assert terminated_track.detection_ids == ("d0",)
    assert tentative_track.detection_ids == ("d1",)
