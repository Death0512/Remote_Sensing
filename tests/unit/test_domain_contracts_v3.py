from dataclasses import FrozenInstanceError, is_dataclass
from datetime import datetime, timezone
import pytest
from shapely.geometry import Point, Polygon
from src.domain import (
    Alert,
    AlertPriority,
    Association,
    AxisAlignedBoundingBox,
    CalibrationState,
    Candidate,
    Capability,
    ConfidenceVector,
    Decision,
    Detection,
    Evidence,
    EvidenceKind,
    GeometricState,
    GeodeticGeometry,
    GeometryStatus,
    Hypothesis,
    HypothesisStatement,
    MapGeometry,
    Measurement,
    MotionClass,
    Observation,
    OrientedBoundingBox,
    PixelGeometry,
    ProcessedObservation,
    Provenance,
    QualityAssessment,
    QualityTier,
    RadiometricState,
    ReviewAction,
    ReviewState,
    ScoreMethod,
    SensorType,
    TargetClass,
    Track,
    TrackLifecycle,
    Zone,
)


def _sample_provenance() -> Provenance:
    return Provenance(
        provenance_id="prov-v3-001",
        source_asset_id="asset-v3-001",
        acquisition_metadata_ref="meta-v3-001",
        pipeline_version="3.0.0",
        git_commit="6f84f43",
        config_hash="sha256:cfgv3",
        model_id=None,
        model_version=None,
        input_checksum="insha256",
        output_checksum="outsha256",
        processing_run_id="run-v3-001",
        processing_time_utc=datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc),
    )


def test_single_detection_contract() -> None:
    assert is_dataclass(Detection)
    field_names = set(Detection.__dataclass_fields__.keys())
    assert "detection_id" in field_names
    assert "candidate_id" in field_names
    assert "sensor" in field_names
    assert "provenance" in field_names


def test_schema_version_present() -> None:
    prov = _sample_provenance()
    dt = datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc)
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    obs = Observation(
        observation_id="obs-1",
        source="CDSE",
        platform="Sentinel-1A",
        sensor=SensorType.S1_SAR,
        product="GRD",
        processing_level="LEVEL1",
        acquisition_time_utc=dt,
        crs="EPSG:4326",
        ground_sample_distance_m=10.0,
        bounds_wgs84=poly,
        asset_uri="s3://asset",
        metadata_uri="s3://meta",
        quality_flags=("OK",),
        provenance=prov,
    )
    assert obs.schema_version == "observation.v3"

    proc = ProcessedObservation(
        processed_id="proc-1",
        observation_id="obs-1",
        preprocessing_profile="sar_profile",
        radiometric_state=RadiometricState.CALIBRATED_SIGMA0,
        geometric_state=GeometricState.TERRAIN_CORRECTED,
        masks=(),
        quality_tier=QualityTier.DATA_OK,
        provenance=prov,
    )
    assert proc.schema_version == "processed_observation.v3"

    meas = Measurement(
        measurement_id="meas-1",
        source_observation_id="obs-1",
        sensor_type=SensorType.S1_SAR,
        value=-12.5,
        unit="dB",
        calibration_status=CalibrationState.PLATT,
        quality_flags=("VALID",),
        provenance=prov,
    )
    assert meas.schema_version == "measurement.v3"

    cand = Candidate(
        candidate_id="cand-1",
        processed_id="proc-1",
        geom_px=(10.0, 10.0, 20.0, 20.0),
        geom_wgs84=Point(108.0, 16.0),
        generator="local_cfar",
        generator_capability=Capability.IMPLEMENTED,
        features=(("snr", 15.0),),
        quality_flags=(),
        provenance=prov,
    )
    assert cand.schema_version == "candidate.v3"

    det = Detection(
        detection_id="det-1",
        candidate_id="cand-1",
        sensor=SensorType.S1_SAR,
        geom_wgs84=Point(108.0, 16.0),
        geom_px=(10.0, 10.0, 20.0, 20.0),
        crs="EPSG:4326",
        target_class=TargetClass.VESSEL,
        confidence=ConfidenceVector(raw_score=0.9),
        attributes=(),
        acquisition_time_utc=dt,
        provenance=prov,
    )
    assert det.schema_version == "detection.v3"

    ev = Evidence(
        evidence_id="ev-1",
        kind=EvidenceKind.OBJECT_CONFIDENCE,
        value=0.9,
        quality=1.0,
        source="sar_cfar",
        timestamp_utc=dt,
        uncertainty=0.05,
        supports="vessel_target",
        references=("det-1",),
        provenance=prov,
    )
    assert ev.schema_version == "evidence.v3"

    assoc = Association(
        association_id="assoc-1",
        source_id="det-1",
        target_id="trk-1",
        association_type="MAHALANOBIS",
        residual=1.2,
        gate_statistic=2.4,
        confidence=0.88,
        provenance=prov,
    )
    assert assoc.schema_version == "association.v3"

    trk = Track(
        track_id="trk-1",
        detection_ids=("det-1",),
        state_estimates=((("x", 1.0), ("y", 2.0)),),
        track_confidence=0.9,
        lifecycle=TrackLifecycle.CONFIRMED,
        provenance=prov,
    )
    assert trk.schema_version == "track.v3"

    hyp = Hypothesis(
        hypothesis_id="hyp-1",
        statement=HypothesisStatement.DARK_VESSEL_CANDIDATE,
        evidence_ids=("ev-1",),
        supporting_evidence=("ev-1",),
        refuting_evidence=(),
        score=0.9,
        score_method=ScoreMethod.RULE_ENGINE,
        calibration_state=CalibrationState.NOT_CALIBRATED,
        explanation="SAR vessel detection with zero correlated AIS.",
        provenance=prov,
    )
    assert hyp.schema_version == "hypothesis.v3"

    dec = Decision(
        decision_id="dec-1",
        action=ReviewAction.CONFIRM,
        rationale="Analyst confirmed dark vessel candidate",
        timestamp_utc=dt,
        provenance=prov,
    )
    assert dec.schema_version == "decision.v3"

    alt = Alert(
        alert_id="alt-1",
        hypothesis_id="hyp-1",
        zone=Zone.ZONE_1,
        priority=AlertPriority.HIGH,
        review_state=ReviewState.PENDING,
        bundle_uri="s3://alerts/alt-1",
        provenance=prov,
    )
    assert alt.schema_version == "alert.v3"
    assert prov.schema_version == "provenance.v3"

    qa = QualityAssessment(tier=QualityTier.DATA_OK, score=0.95, flags=("CALIBRATED",))
    assert qa.schema_version == "quality.v3"


def test_no_fabricated_probability() -> None:
    vec_uncalibrated = ConfidenceVector(raw_score=0.85)
    assert vec_uncalibrated.calibrated_probability is None
    assert vec_uncalibrated.calibration_state == CalibrationState.NOT_CALIBRATED

    with pytest.raises(ValueError, match="calibrated_probability must be None"):
        ConfidenceVector(
            raw_score=0.85,
            calibrated_probability=0.85,
            calibration_state=CalibrationState.NOT_CALIBRATED,
        )

    vec_calibrated = ConfidenceVector(
        raw_score=0.85,
        calibrated_probability=0.82,
        calibration_state=CalibrationState.PLATT,
    )
    assert vec_calibrated.calibrated_probability == 0.82
    assert vec_calibrated.calibration_state == CalibrationState.PLATT

    prov = _sample_provenance()
    with pytest.raises(ValueError, match="calibrated_probability must be None"):
        Hypothesis(
            hypothesis_id="hyp-fail",
            statement=HypothesisStatement.DARK_VESSEL_CANDIDATE,
            evidence_ids=("ev-1",),
            supporting_evidence=("ev-1",),
            refuting_evidence=(),
            score=0.8,
            score_method=ScoreMethod.RULE_ENGINE,
            calibration_state=CalibrationState.NOT_CALIBRATED,
            calibrated_probability=0.8,
            explanation="Invalid uncalibrated probability",
            provenance=prov,
        )


def test_provenance_required() -> None:
    dt = datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc)
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    with pytest.raises((TypeError, ValueError)):
        Observation(
            observation_id="obs-1",
            source="CDSE",
            platform="Sentinel-1A",
            sensor=SensorType.S1_SAR,
            product="GRD",
            processing_level="LEVEL1",
            acquisition_time_utc=dt,
            crs="EPSG:4326",
            ground_sample_distance_m=10.0,
            bounds_wgs84=poly,
            asset_uri="s3://asset",
            metadata_uri="s3://meta",
            quality_flags=("OK",),
            provenance=None,
        )

    with pytest.raises((TypeError, ValueError)):
        ProcessedObservation(
            processed_id="proc-1",
            observation_id="obs-1",
            preprocessing_profile="sar",
            radiometric_state=RadiometricState.CALIBRATED_SIGMA0,
            geometric_state=GeometricState.TERRAIN_CORRECTED,
            masks=(),
            quality_tier=QualityTier.DATA_OK,
            provenance=None,
        )

    with pytest.raises((TypeError, ValueError)):
        Measurement(
            measurement_id="meas-1",
            source_observation_id="obs-1",
            sensor_type=SensorType.S1_SAR,
            value=10.0,
            unit="dB",
            calibration_status=CalibrationState.PLATT,
            quality_flags=(),
            provenance=None,
        )

    with pytest.raises((TypeError, ValueError)):
        Candidate(
            candidate_id="cand-1",
            processed_id="proc-1",
            geom_px=(0, 0, 1, 1),
            geom_wgs84=Point(0, 0),
            generator="cfar",
            generator_capability=Capability.IMPLEMENTED,
            features=(),
            quality_flags=(),
            provenance=None,
        )

    with pytest.raises((TypeError, ValueError)):
        Detection(
            detection_id="det-1",
            candidate_id="cand-1",
            sensor=SensorType.S1_SAR,
            geom_wgs84=Point(0, 0),
            geom_px=(0, 0, 1, 1),
            crs="EPSG:4326",
            target_class=TargetClass.VESSEL,
            confidence=ConfidenceVector(raw_score=0.9),
            attributes=(),
            acquisition_time_utc=dt,
            provenance=None,
        )

    with pytest.raises((TypeError, ValueError)):
        Evidence(
            evidence_id="ev-1",
            kind=EvidenceKind.OBJECT_CONFIDENCE,
            value=0.9,
            quality=1.0,
            source="cfar",
            timestamp_utc=dt,
            uncertainty=None,
            supports="target",
            references=(),
            provenance=None,
        )

    with pytest.raises((TypeError, ValueError)):
        Association(
            association_id="assoc-1",
            source_id="d1",
            target_id="t1",
            association_type="MAHALANOBIS",
            residual=1.0,
            gate_statistic=1.0,
            confidence=0.9,
            provenance=None,
        )

    with pytest.raises((TypeError, ValueError)):
        Track(
            track_id="trk-1",
            detection_ids=("d1",),
            state_estimates=(),
            track_confidence=0.9,
            lifecycle=TrackLifecycle.CONFIRMED,
            provenance=None,
        )

    with pytest.raises((TypeError, ValueError)):
        Hypothesis(
            hypothesis_id="hyp-1",
            statement=HypothesisStatement.DARK_VESSEL_CANDIDATE,
            evidence_ids=("ev-1",),
            supporting_evidence=("ev-1",),
            refuting_evidence=(),
            score=0.9,
            score_method=ScoreMethod.RULE_ENGINE,
            calibration_state=CalibrationState.NOT_CALIBRATED,
            explanation="Explanation",
            provenance=None,
        )

    with pytest.raises((TypeError, ValueError)):
        Decision(
            decision_id="dec-1",
            action=ReviewAction.CONFIRM,
            rationale="Rationale",
            timestamp_utc=dt,
            provenance=None,
        )

    with pytest.raises((TypeError, ValueError)):
        Alert(
            alert_id="alt-1",
            hypothesis_id="hyp-1",
            zone=Zone.ZONE_1,
            priority=AlertPriority.HIGH,
            review_state=ReviewState.PENDING,
            bundle_uri="s3://alert",
            provenance=None,
        )


def test_geometry_status_enforced() -> None:
    expected_statuses = {"GEOCODED", "MAP_ONLY", "PIXEL_ONLY", "UNKNOWN"}
    assert {s.value for s in GeometryStatus} == expected_statuses
    assert GeometryStatus.GEOCODED == "GEOCODED"
    assert GeometryStatus.GEOCODED == "geocoded"
    assert GeometryStatus("geocoded") is GeometryStatus.GEOCODED
    assert GeometryStatus("GEOCODED") is GeometryStatus.GEOCODED

    px = PixelGeometry(row=10.5, col=20.5, shape=(512, 512))
    assert px.row == 10.5
    assert px.col == 20.5
    assert px.shape == (512, 512)

    mg = MapGeometry(
        x=500000.0,
        y=1800000.0,
        crs="EPSG:32648",
        affine=(10.0, 0.0, 500000.0, 0.0, -10.0, 1800000.0),
    )
    assert mg.crs == "EPSG:32648"
    assert mg.affine[0] == 10.0

    geo = GeodeticGeometry(lon=108.0, lat=16.0, crs="EPSG:4326", horizontal_accuracy_m=5.0)
    assert geo.lon == 108.0
    assert geo.lat == 16.0
    assert geo.horizontal_accuracy_m == 5.0

    assert type(px) is not type(geo)
    assert type(px) is not type(mg)
    assert type(mg) is not type(geo)

    prov = _sample_provenance()
    dt = datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc)

    meas = Measurement(
        measurement_id="m1",
        source_observation_id="o1",
        sensor_type=SensorType.S1_SAR,
        value=-15.0,
        unit="dB",
        calibration_status=CalibrationState.PLATT,
        quality_flags=(),
        provenance=prov,
        geometry=geo,
        geometry_status=GeometryStatus.GEOCODED,
    )
    assert meas.geometry_status == GeometryStatus.GEOCODED
    assert meas.geometry == geo

    cand_pixel = Candidate(
        candidate_id="c1",
        processed_id="p1",
        geom_px=(10, 10, 20, 20),
        geom_wgs84=Point(0, 0),
        generator="cfar",
        generator_capability=Capability.IMPLEMENTED,
        features=(),
        quality_flags=(),
        provenance=prov,
        geometry_status=GeometryStatus.PIXEL_ONLY,
    )
    assert cand_pixel.geometry_status == GeometryStatus.PIXEL_ONLY


def test_immutability() -> None:
    prov = _sample_provenance()
    dt = datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc)

    meas = Measurement(
        measurement_id="meas-1",
        source_observation_id="obs-1",
        sensor_type=SensorType.S1_SAR,
        value=-12.5,
        unit="dB",
        calibration_status=CalibrationState.NOT_CALIBRATED,
        quality_flags=(),
        provenance=prov,
    )
    with pytest.raises(FrozenInstanceError):
        meas.value = 0.0

    dec = Decision(
        decision_id="dec-1",
        action=ReviewAction.CONFIRM,
        rationale="Verified target",
        timestamp_utc=dt,
        provenance=prov,
    )
    with pytest.raises(FrozenInstanceError):
        dec.action = ReviewAction.DISMISS

    assoc = Association(
        association_id="assoc-1",
        source_id="d1",
        target_id="t1",
        association_type="MAHALANOBIS",
        residual=1.0,
        gate_statistic=2.0,
        confidence=0.85,
        provenance=prov,
    )
    with pytest.raises(FrozenInstanceError):
        assoc.confidence = 0.5

    px = PixelGeometry(row=1.0, col=2.0, shape=(10, 10))
    with pytest.raises(FrozenInstanceError):
        px.row = 5.0

    mg = MapGeometry(x=1.0, y=2.0, crs="EPSG:32648", affine=(1, 0, 0, 0, 1, 0))
    with pytest.raises(FrozenInstanceError):
        mg.crs = "EPSG:4326"

    geo = GeodeticGeometry(lon=108.0, lat=16.0)
    with pytest.raises(FrozenInstanceError):
        geo.lon = 0.0


def test_decision_rationale_required() -> None:
    prov = _sample_provenance()
    dt = datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="rationale must not be empty"):
        Decision(
            decision_id="dec-1",
            action=ReviewAction.CONFIRM,
            rationale="",
            timestamp_utc=dt,
            provenance=prov,
        )

    with pytest.raises(ValueError, match="rationale must not be empty"):
        Decision(
            decision_id="dec-1",
            action=ReviewAction.CONFIRM,
            rationale="   ",
            timestamp_utc=dt,
            provenance=prov,
        )


def test_obb_and_aabb_domain_contracts() -> None:
    aabb = AxisAlignedBoundingBox(min_col=10.0, min_row=15.0, max_col=30.0, max_row=45.0)
    assert aabb.width == 20.0
    assert aabb.height == 30.0
    assert aabb.cx == 20.0
    assert aabb.cy == 30.0
    assert aabb.to_tuple() == (10.0, 15.0, 30.0, 45.0)

    obb = OrientedBoundingBox(cx=20.0, cy=30.0, width=20.0, height=30.0, angle_deg=0.0)
    assert obb.cx == 20.0
    assert obb.cy == 30.0
    assert obb.to_tuple() == (20.0, 30.0, 20.0, 30.0, 0.0)
    assert obb.to_aabb() == aabb

