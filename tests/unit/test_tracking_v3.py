from datetime import datetime, timedelta, timezone
import numpy as np
import pytest
from shapely.geometry import Point

from src.domain.confidence import ConfidenceVector
from src.domain.detection import Detection
from src.domain.enums import Capability, MotionClass, SensorType, TargetClass, TrackLifecycle
from src.domain.provenance import Provenance
from src.domain.track import Track
from src.provenance.engine import compute_bytes_checksum, get_git_commit
from src.tracking.association.hungarian import MahalanobisAssociator
from src.tracking.filters.kalman import KalmanFilter2D
from src.tracking.motion_models import (
    AIRCRAFT_MODEL,
    MOTION_MODELS,
    STATIONARY_MODEL,
    UNKNOWN_MODEL,
    VESSEL_MODEL,
    MotionModelConfig,
    get_motion_model,
    target_class_to_motion_class,
)
from src.tracking.tracker import MultiLookTracker, TrackState


def _make_provenance(prov_id: str = "prov-t3") -> Provenance:
    return Provenance(
        provenance_id=prov_id,
        source_asset_id="asset-trk3",
        acquisition_metadata_ref="s3://meta/trk3.xml",
        pipeline_version="3.0.0",
        git_commit=get_git_commit(),
        config_hash=compute_bytes_checksum(b"cfg-test-v3"),
        model_id=None,
        model_version=None,
        input_checksum=compute_bytes_checksum(b"input-bytes"),
        output_checksum=compute_bytes_checksum(b"output-bytes"),
        processing_run_id="run-trk3",
        processing_time_utc=datetime(2026, 10, 23, 10, 0, 0, tzinfo=timezone.utc),
    )


def _make_detection(
    det_id: str,
    lon: float,
    lat: float,
    acq_time: datetime,
    target_class: TargetClass = TargetClass.VESSEL,
    sensor: SensorType = SensorType.S1_SAR,
    time_uncertainty_seconds: float = 0.0,
    measurement_noise_std: float | None = None,
) -> Detection:
    attrs: list[tuple[str, float]] = []
    if time_uncertainty_seconds > 0.0:
        attrs.append(("time_uncertainty_seconds", time_uncertainty_seconds))
    if measurement_noise_std is not None:
        attrs.append(("measurement_noise_std", measurement_noise_std))

    return Detection(
        detection_id=det_id,
        candidate_id=f"cand-{det_id}",
        sensor=sensor,
        geom_wgs84=Point(lon, lat),
        geom_px=(100.0, 100.0, 120.0, 110.0),
        crs="EPSG:4326",
        target_class=target_class,
        confidence=ConfidenceVector(raw_score=0.92),
        attributes=tuple(attrs),
        acquisition_time_utc=acq_time,
        provenance=_make_provenance(f"prov-{det_id}"),
    )


def test_kalman_predict_update() -> None:
    vessel_model = get_motion_model(MotionClass.VESSEL)
    kf = KalmanFilter2D(
        initial_state=np.array([0.0, 0.0, 5.0, 2.0]),
        initial_covariance=np.diag([25.0, 25.0, 25.0, 25.0]),
        motion_model=vessel_model,
    )

    pred_state, pred_cov = kf.predict(10.0)
    assert abs(pred_state[0, 0] - 50.0) < 1e-3
    assert abs(pred_state[1, 0] - 20.0) < 1e-3
    assert pred_cov[0, 0] > 25.0
    assert pred_cov[1, 1] > 25.0

    upd_state, upd_cov = kf.update(np.array([51.0, 19.5]))
    assert upd_cov[0, 0] < pred_cov[0, 0]
    assert upd_cov[1, 1] < pred_cov[1, 1]
    assert 45.0 <= upd_state[0, 0] <= 55.0
    assert 15.0 <= upd_state[1, 0] <= 25.0

    aircraft_model = get_motion_model(MotionClass.AIRCRAFT)
    kf_air = KalmanFilter2D(
        initial_state=np.array([1000.0, 2000.0, 150.0, 0.0]),
        motion_model=aircraft_model,
    )
    air_state, _ = kf_air.predict(2.0)
    assert abs(air_state[0, 0] - 1300.0) < 1e-3


def test_mahalanobis_gate() -> None:
    associator = MahalanobisAssociator(default_chi2_threshold=9.21034)

    class FakeTrack:
        def __init__(self, x: float, y: float, vx: float, vy: float, model: MotionModelConfig):
            self.predicted_pos = (x, y)
            self.velocity = (vx, vy)
            self.covariance = np.diag([25.0, 25.0, 16.0, 16.0])
            self.motion_model = model

    class FakeDetection:
        def __init__(self, x: float, y: float, meas_std: float = 5.0):
            self.pos = (x, y)
            self.measurement_covariance = np.diag([meas_std**2, meas_std**2])

    vessel_model = get_motion_model(MotionClass.VESSEL)
    track1 = FakeTrack(0.0, 0.0, 0.0, 0.0, vessel_model)
    det_inlier = FakeDetection(3.0, 4.0, meas_std=5.0)
    det_outlier = FakeDetection(300.0, 400.0, meas_std=5.0)

    matched, unmatched_trks, unmatched_dets = associator.associate(
        [track1],
        [det_inlier, det_outlier],
    )
    assert matched == [(0, 0)]
    assert unmatched_trks == []
    assert unmatched_dets == [1]

    det_a = FakeDetection(2.0, 2.0, meas_std=5.0)
    det_b = FakeDetection(4.0, 4.0, meas_std=5.0)
    matched_dup, unmatched_trks_dup, unmatched_dets_dup = associator.associate(
        [track1],
        [det_a, det_b],
    )
    assert matched_dup == [(0, 0)]
    assert unmatched_dets_dup == [1]


def test_motion_class_distinct() -> None:
    assert STATIONARY_MODEL.motion_class == MotionClass.STATIONARY
    assert VESSEL_MODEL.motion_class == MotionClass.VESSEL
    assert AIRCRAFT_MODEL.motion_class == MotionClass.AIRCRAFT
    assert UNKNOWN_MODEL.motion_class == MotionClass.UNKNOWN

    assert STATIONARY_MODEL.process_noise_vel == 0.0
    assert VESSEL_MODEL.process_noise_vel == 0.5
    assert AIRCRAFT_MODEL.process_noise_vel == 10.0
    assert UNKNOWN_MODEL.process_noise_vel == 2.0

    assert STATIONARY_MODEL.max_speed_mps == 0.5
    assert VESSEL_MODEL.max_speed_mps < 25.0
    assert AIRCRAFT_MODEL.max_speed_mps > 300.0

    assert STATIONARY_MODEL.max_dt_seconds == 86400.0
    assert VESSEL_MODEL.max_dt_seconds == 7200.0
    assert AIRCRAFT_MODEL.max_dt_seconds == 1800.0
    assert UNKNOWN_MODEL.max_dt_seconds == 3600.0

    kf_stat = KalmanFilter2D(
        initial_state=np.array([100.0, 200.0, 10.0, 10.0]),
        motion_model=STATIONARY_MODEL,
    )
    assert kf_stat.state[2, 0] == 0.0
    assert kf_stat.state[3, 0] == 0.0
    stat_pred, _ = kf_stat.predict(60.0)
    assert stat_pred[0, 0] == 100.0
    assert stat_pred[1, 0] == 200.0
    assert stat_pred[2, 0] == 0.0
    assert stat_pred[3, 0] == 0.0

    kf_vessel = KalmanFilter2D(
        initial_state=np.array([0.0, 0.0, 50.0, 0.0]),
        motion_model=VESSEL_MODEL,
    )
    assert kf_vessel.state[2, 0] <= VESSEL_MODEL.max_speed_mps + 1e-4

    kf_air = KalmanFilter2D(
        initial_state=np.array([0.0, 0.0, 250.0, 0.0]),
        motion_model=AIRCRAFT_MODEL,
    )
    assert kf_air.state[2, 0] == 250.0


def test_time_uncertainty_propagates() -> None:
    associator = MahalanobisAssociator(default_chi2_threshold=9.21034)

    p_pred = np.diag([25.0, 25.0, 0.0, 0.0])
    r_meas = np.diag([25.0, 25.0])
    velocity = (50.0, 0.0)

    s_zero = associator.compute_innovation_covariance(
        p_pred=p_pred,
        r_meas=r_meas,
        velocity=velocity,
        sigma_t=0.0,
    )
    assert s_zero[0, 0] == 50.0
    assert s_zero[1, 1] == 50.0
    assert s_zero[0, 1] == 0.0

    s_time = associator.compute_innovation_covariance(
        p_pred=p_pred,
        r_meas=r_meas,
        velocity=velocity,
        sigma_t=1.0,
    )
    assert s_time[0, 0] == 50.0 + 2500.0
    assert s_time[1, 1] == 50.0
    assert s_time[0, 1] == 0.0

    residual = np.array([[25.0], [0.0]], dtype=np.float64)
    d2_zero = associator.compute_mahalanobis_distance_sq(residual, s_zero)
    d2_time = associator.compute_mahalanobis_distance_sq(residual, s_time)

    assert d2_zero == pytest.approx(12.5, abs=0.01)
    assert d2_zero > 9.21034

    assert d2_time == pytest.approx(625.0 / 2550.0, abs=0.01)
    assert d2_time < 9.21034


def test_delta_t_max_bounds_per_class() -> None:
    tracker = MultiLookTracker(center_lat=16.0, center_lon=108.0)
    t0 = datetime(2026, 10, 23, 10, 0, 0, tzinfo=timezone.utc)

    d_vessel_0 = _make_detection("v0", 108.0, 16.0, t0, target_class=TargetClass.VESSEL)
    d_air_0 = _make_detection("a0", 108.1, 16.1, t0, target_class=TargetClass.AIRCRAFT)
    d_stat_0 = _make_detection("s0", 108.2, 16.2, t0, target_class=TargetClass.AIRCRAFT_PARKED)

    tracker.update([d_vessel_0, d_air_0, d_stat_0], t0)
    assert len(tracker.active_tracks) == 3

    t_confirm = t0 + timedelta(minutes=2)
    d_vessel_1 = _make_detection("v1", 108.0001, 16.0001, t_confirm, target_class=TargetClass.VESSEL)
    d_air_1 = _make_detection("a1", 108.1001, 16.1001, t_confirm, target_class=TargetClass.AIRCRAFT)
    d_stat_1 = _make_detection("s1", 108.20001, 16.20001, t_confirm, target_class=TargetClass.AIRCRAFT_PARKED)

    conf_tracks = tracker.update([d_vessel_1, d_air_1, d_stat_1], t_confirm)
    assert all(t.lifecycle == TrackLifecycle.CONFIRMED for t in conf_tracks)

    t_45min = t_confirm + timedelta(minutes=45)
    trks_45 = tracker.update([], t_45min)
    air_trks = [t for t in trks_45 if t.motion_class == MotionClass.AIRCRAFT]
    assert len(air_trks) == 1
    assert air_trks[0].lifecycle == TrackLifecycle.TERMINATED

    vessel_trks = [t for t in trks_45 if t.motion_class == MotionClass.VESSEL]
    assert len(vessel_trks) == 1
    assert vessel_trks[0].lifecycle == TrackLifecycle.COASTING

    stat_trks = [t for t in trks_45 if t.motion_class == MotionClass.STATIONARY]
    assert len(stat_trks) == 1
    assert stat_trks[0].lifecycle == TrackLifecycle.COASTING

    t_3hours = t_confirm + timedelta(hours=3)
    trks_3h = tracker.update([], t_3hours)
    vessel_trks_3h = [t for t in trks_3h if t.motion_class == MotionClass.VESSEL]
    assert len(vessel_trks_3h) == 1
    assert vessel_trks_3h[0].lifecycle == TrackLifecycle.TERMINATED

    stat_trks_3h = [t for t in trks_3h if t.motion_class == MotionClass.STATIONARY]
    assert len(stat_trks_3h) == 1
    assert stat_trks_3h[0].lifecycle == TrackLifecycle.COASTING


def test_canonical_v3_track_contract() -> None:
    tracker = MultiLookTracker(center_lat=16.0, center_lon=108.0)
    t0 = datetime(2026, 10, 23, 10, 0, 0, tzinfo=timezone.utc)
    d0 = _make_detection("d0", 108.0, 16.0, t0, target_class=TargetClass.VESSEL)

    tracks = tracker.update([d0], t0)
    assert len(tracks) == 1
    tr = tracks[0]

    assert isinstance(tr, Track)
    assert tr.schema_version == "track.v3"
    assert tr.motion_class == MotionClass.VESSEL
    assert isinstance(tr.state, tuple)
    assert len(tr.state) == 4
    assert isinstance(tr.state_covariance, tuple)
    assert len(tr.state_covariance) == 4
    assert all(len(row) == 4 for row in tr.state_covariance)
    assert isinstance(tr.last_update_utc, datetime)
    assert isinstance(tr.prediction_time_utc, datetime)
    assert tr.associated_detection_ids == ("d0",)
    assert tr.sensor_sources == (SensorType.S1_SAR.value,)
    assert 0.0 <= tr.quality <= 1.0
    assert tr.provenance.git_commit != "unknown"
    assert len(tr.provenance.config_hash) == 64
    assert tr.lifecycle == TrackLifecycle.TENTATIVE
