from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
from typing import Any, Optional, Sequence, Union
import numpy as np
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from src.benchmark.contracts import DatasetSplit, GoldenSample, GroundTruthTarget
from src.benchmark.metrics import extract_geometry
from src.domain.enums import QualityTier, SensorType, TargetClass
from src.domain.observation import Observation, ProcessedObservation
from src.domain.provenance import Provenance
from src.preprocessing.optical.preprocessor import OpticalPreprocessor


class LeakageError(Exception):
    pass


class SpatialLeakageError(LeakageError):
    pass


class TemporalLeakageError(LeakageError):
    pass


def validate_no_leakage(
    train_samples: Sequence[GoldenSample],
    test_samples: Sequence[GoldenSample],
    guard_window_seconds: float = 0.0,
) -> bool:
    for s_train in train_samples:
        train_geom = extract_geometry(s_train.bounds_wgs84)
        for s_test in test_samples:
            test_geom = extract_geometry(s_test.bounds_wgs84)
            if train_geom.intersects(test_geom):
                inter = train_geom.intersection(test_geom)
                if inter.area > 0.0:
                    raise SpatialLeakageError(
                        f"Spatial leakage detected between train sample {s_train.sample_id} and test sample {s_test.sample_id}: intersection area = {inter.area}"
                    )

    guard_delta = timedelta(seconds=max(0.0, float(guard_window_seconds)))
    for s_train in train_samples:
        t_tr = s_train.acquisition_time_utc
        tr_start = s_train.metadata.get("start_time_utc", t_tr)
        tr_end = s_train.metadata.get("end_time_utc", t_tr)
        if tr_start > tr_end:
            tr_start, tr_end = tr_end, tr_start
        tr_start_guard = tr_start - guard_delta
        tr_end_guard = tr_end + guard_delta

        for s_test in test_samples:
            t_te = s_test.acquisition_time_utc
            te_start = s_test.metadata.get("start_time_utc", t_te)
            te_end = s_test.metadata.get("end_time_utc", t_te)
            if te_start > te_end:
                te_start, te_end = te_end, te_start

            latest_start = max(tr_start_guard, te_start)
            earliest_end = min(tr_end_guard, te_end)
            if latest_start <= earliest_end:
                raise TemporalLeakageError(
                    f"Temporal leakage detected between train sample {s_train.sample_id} [{tr_start.isoformat()} - {tr_end.isoformat()}] and test sample {s_test.sample_id} [{te_start.isoformat()} - {te_end.isoformat()}] with guard window {guard_window_seconds}s"
                )

    return True


@dataclass(frozen=True)
class GoldenDataset:
    name: str
    train_samples: tuple[GoldenSample, ...] = ()
    validation_samples: tuple[GoldenSample, ...] = ()
    test_samples: tuple[GoldenSample, ...] = ()
    guard_window_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, guard_window_seconds: Optional[float] = None) -> bool:
        gw = self.guard_window_seconds if guard_window_seconds is None else guard_window_seconds
        if self.train_samples and self.validation_samples:
            validate_no_leakage(self.train_samples, self.validation_samples, guard_window_seconds=gw)
        if self.train_samples and self.test_samples:
            validate_no_leakage(self.train_samples, self.test_samples, guard_window_seconds=gw)
        if self.validation_samples and self.test_samples:
            validate_no_leakage(self.validation_samples, self.test_samples, guard_window_seconds=gw)
        return True

    def get_split(self, split: DatasetSplit) -> tuple[GoldenSample, ...]:
        if split == DatasetSplit.TRAIN:
            return self.train_samples
        if split == DatasetSplit.VALIDATION:
            return self.validation_samples
        if split == DatasetSplit.TEST:
            return self.test_samples
        raise ValueError(f"Unknown split: {split}")

    def all_samples(self) -> tuple[GoldenSample, ...]:
        return self.train_samples + self.validation_samples + self.test_samples


def _compute_sha256(data: Any) -> str:
    if isinstance(data, np.ndarray):
        return hashlib.sha256(data.tobytes()).hexdigest()
    if isinstance(data, tuple):
        h = hashlib.sha256()
        for elem in data:
            if isinstance(elem, np.ndarray):
                h.update(elem.tobytes())
            elif isinstance(elem, bytes):
                h.update(elem)
            else:
                h.update(str(elem).encode("utf-8"))
        return h.hexdigest()
    if isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()
    return hashlib.sha256(str(data).encode("utf-8")).hexdigest()


def _make_provenance(prov_id: str, asset_id: str) -> Provenance:
    return Provenance(
        provenance_id=prov_id,
        source_asset_id=asset_id,
        acquisition_metadata_ref=f"s3://golden-benchmarks/{asset_id}.xml",
        pipeline_version="3.0.0",
        git_commit="3536f55",
        config_hash="cfg-golden-benchmark-v3",
        model_id=None,
        model_version=None,
        input_checksum=hashlib.sha256(asset_id.encode("utf-8")).hexdigest()[:16],
        output_checksum=hashlib.sha256((asset_id + ":proc").encode("utf-8")).hexdigest()[:16],
        processing_run_id=f"run-{prov_id}",
        processing_time_utc=datetime.now(timezone.utc),
    )


def create_sar_vessel_golden_dataset(num_samples_per_split: int = 1) -> GoldenDataset:
    splits_config = [
        (DatasetSplit.TRAIN, datetime(2026, 1, 10, 0, 0, 0, tzinfo=timezone.utc), (108.0, 15.0, 108.2, 15.2)),
        (DatasetSplit.VALIDATION, datetime(2026, 5, 10, 0, 0, 0, tzinfo=timezone.utc), (110.0, 17.0, 110.2, 17.2)),
        (DatasetSplit.TEST, datetime(2026, 10, 10, 0, 0, 0, tzinfo=timezone.utc), (112.0, 19.0, 112.2, 19.2)),
    ]

    split_samples: dict[DatasetSplit, list[GoldenSample]] = {
        DatasetSplit.TRAIN: [],
        DatasetSplit.VALIDATION: [],
        DatasetSplit.TEST: [],
    }

    for split, base_time, (min_lon, min_lat, max_lon, max_lat) in splits_config:
        for idx in range(num_samples_per_split):
            sid = f"sar-vessel-{split.value}-{idx + 1}"
            cur_min_lon = min_lon + idx * 0.3
            cur_max_lon = max_lon + idx * 0.3
            cur_time = base_time + timedelta(days=idx * 2)

            h, w = 120, 120
            img = np.ones((h, w), dtype=np.float32) * 5.0
            r_start, r_end = 40, 48
            c_start, c_end = 40, 43
            img[r_start:r_end, c_start:c_end] = 300.0

            res_lon = (cur_max_lon - cur_min_lon) / w
            res_lat = (max_lat - min_lat) / h
            t_min_lon = cur_min_lon + c_start * res_lon
            t_max_lon = cur_min_lon + c_end * res_lon
            t_min_lat = max_lat - r_end * res_lat
            t_max_lat = max_lat - r_start * res_lat

            target = GroundTruthTarget(
                target_id=f"tgt-{sid}-1",
                target_class=TargetClass.VESSEL,
                geom_wgs84=box(t_min_lon, t_min_lat, t_max_lon, t_max_lat),
                geom_px=(float(c_start), float(r_start), float(c_end), float(r_end)),
                properties={"length_m": 80.0, "width_m": 30.0, "snr": 60.0},
            )

            bounds_geom = box(cur_min_lon, min_lat, cur_max_lon, max_lat)
            sample = GoldenSample(
                sample_id=sid,
                dataset_name="golden-sar-vessel-v3",
                source="Copernicus Sentinel-1 SAR GRD",
                license="CC-BY-4.0 / CDSE Open Access",
                acquisition_time_utc=cur_time,
                sensor=SensorType.S1_SAR,
                crs="EPSG:4326",
                gsd_m=10.0,
                bounds_wgs84=bounds_geom,
                ground_truth=(target,),
                checksum=_compute_sha256(img),
                split=split,
                metadata={"polarization": "VV", "orbit": "DESCENDING"},
                image_data=img,
                land_mask=np.zeros((h, w), dtype=bool),
                area_km2=float((h * 10.0 * w * 10.0) / 1_000_000.0),
            )
            split_samples[split].append(sample)

    dataset = GoldenDataset(
        name="golden-sar-vessel-v3",
        train_samples=tuple(split_samples[DatasetSplit.TRAIN]),
        validation_samples=tuple(split_samples[DatasetSplit.VALIDATION]),
        test_samples=tuple(split_samples[DatasetSplit.TEST]),
        guard_window_seconds=86400.0 * 30.0,
    )
    dataset.validate()
    return dataset


def create_optical_aircraft_golden_dataset(num_samples_per_split: int = 1) -> GoldenDataset:
    splits_config = [
        (DatasetSplit.TRAIN, datetime(2026, 1, 15, 0, 0, 0, tzinfo=timezone.utc), (100.0, 10.0, 100.1, 10.1)),
        (DatasetSplit.VALIDATION, datetime(2026, 5, 15, 0, 0, 0, tzinfo=timezone.utc), (102.0, 12.0, 102.1, 12.1)),
        (DatasetSplit.TEST, datetime(2026, 10, 15, 0, 0, 0, tzinfo=timezone.utc), (104.0, 14.0, 104.1, 14.1)),
    ]

    split_samples: dict[DatasetSplit, list[GoldenSample]] = {
        DatasetSplit.TRAIN: [],
        DatasetSplit.VALIDATION: [],
        DatasetSplit.TEST: [],
    }

    preprocessor = OpticalPreprocessor()

    for split, base_time, (min_lon, min_lat, max_lon, max_lat) in splits_config:
        for idx in range(num_samples_per_split):
            sid = f"opt-aircraft-{split.value}-{idx + 1}"
            cur_min_lon = min_lon + idx * 0.2
            cur_max_lon = max_lon + idx * 0.2
            cur_time = base_time + timedelta(days=idx * 2)

            h, w = 100, 100
            img = np.zeros((h, w), dtype=np.float32)
            r_start, r_end = 35, 50
            c_start, c_end = 35, 42
            img[r_start:r_end, c_start:c_end] = 0.95

            res_lon = (cur_max_lon - cur_min_lon) / w
            res_lat = (max_lat - min_lat) / h
            t_min_lon = cur_min_lon + c_start * res_lon
            t_max_lon = cur_min_lon + c_end * res_lon
            t_min_lat = max_lat - r_end * res_lat
            t_max_lat = max_lat - r_start * res_lat

            target = GroundTruthTarget(
                target_id=f"tgt-{sid}-1",
                target_class=TargetClass.AIRCRAFT,
                geom_wgs84=box(t_min_lon, t_min_lat, t_max_lon, t_max_lat),
                geom_px=(float(c_start), float(r_start), float(c_end), float(r_end)),
                properties={"length_m": 15.0, "width_m": 7.0},
            )

            bounds_geom = box(cur_min_lon, min_lat, cur_max_lon, max_lat)
            obs = Observation(
                observation_id=f"obs-{sid}",
                source="VHR-Optical-Provider",
                platform="Airborne-EO",
                sensor=SensorType.VHR_OPTICAL,
                product="ORTHO",
                processing_level="L2A",
                acquisition_time_utc=cur_time,
                crs="EPSG:4326",
                ground_sample_distance_m=1.0,
                bounds_wgs84=bounds_geom,
                asset_uri=f"s3://golden-benchmarks/{sid}.tif",
                metadata_uri=f"s3://golden-benchmarks/{sid}.json",
                quality_flags=("CLEAR_SKIES",),
                provenance=_make_provenance(f"prov-{sid}", sid),
            )
            proc_obs = preprocessor.process(obs, cloud_mask=None)

            sample = GoldenSample(
                sample_id=sid,
                dataset_name="golden-optical-aircraft-v3",
                source="VHR Optical Open Datasets",
                license="CC-BY-4.0",
                acquisition_time_utc=cur_time,
                sensor=SensorType.VHR_OPTICAL,
                crs="EPSG:4326",
                gsd_m=1.0,
                bounds_wgs84=bounds_geom,
                ground_truth=(target,),
                checksum=_compute_sha256(img),
                split=split,
                metadata={"cloud_cover": 0.0},
                image_data=img,
                processed_observation=proc_obs,
                area_km2=float((h * 1.0 * w * 1.0) / 1_000_000.0),
            )
            split_samples[split].append(sample)

    dataset = GoldenDataset(
        name="golden-optical-aircraft-v3",
        train_samples=tuple(split_samples[DatasetSplit.TRAIN]),
        validation_samples=tuple(split_samples[DatasetSplit.VALIDATION]),
        test_samples=tuple(split_samples[DatasetSplit.TEST]),
        guard_window_seconds=86400.0 * 30.0,
    )
    dataset.validate()
    return dataset


def create_bitemporal_change_golden_dataset(num_samples_per_split: int = 1) -> GoldenDataset:
    splits_config = [
        (DatasetSplit.TRAIN, datetime(2026, 1, 20, 0, 0, 0, tzinfo=timezone.utc), (105.0, 11.0, 105.1, 11.1)),
        (DatasetSplit.VALIDATION, datetime(2026, 5, 20, 0, 0, 0, tzinfo=timezone.utc), (106.0, 12.0, 106.1, 12.1)),
        (DatasetSplit.TEST, datetime(2026, 10, 20, 0, 0, 0, tzinfo=timezone.utc), (107.0, 13.0, 107.1, 13.1)),
    ]

    split_samples: dict[DatasetSplit, list[GoldenSample]] = {
        DatasetSplit.TRAIN: [],
        DatasetSplit.VALIDATION: [],
        DatasetSplit.TEST: [],
    }

    for split, base_time, (min_lon, min_lat, max_lon, max_lat) in splits_config:
        for idx in range(num_samples_per_split):
            sid = f"change-{split.value}-{idx + 1}"
            cur_min_lon = min_lon + idx * 0.2
            cur_max_lon = max_lon + idx * 0.2
            cur_time = base_time + timedelta(days=idx * 2)

            h, w = 80, 80
            arr_t0 = np.full((h, w), 0.2, dtype=np.float32)
            arr_t1 = arr_t0.copy()
            r_start, r_end = 25, 45
            c_start, c_end = 25, 45
            arr_t1[r_start:r_end, c_start:c_end] = 0.85

            res_lon = (cur_max_lon - cur_min_lon) / w
            res_lat = (max_lat - min_lat) / h
            t_min_lon = cur_min_lon + c_start * res_lon
            t_max_lon = cur_min_lon + c_end * res_lon
            t_min_lat = max_lat - r_end * res_lat
            t_max_lat = max_lat - r_start * res_lat

            target = GroundTruthTarget(
                target_id=f"tgt-{sid}-1",
                target_class=TargetClass.INFRASTRUCTURE_CHANGE,
                geom_wgs84=box(t_min_lon, t_min_lat, t_max_lon, t_max_lat),
                geom_px=(float(c_start), float(r_start), float(c_end), float(r_end)),
                properties={"change_type": "STRUCTURE_APPEARANCE"},
            )

            bounds_geom = box(cur_min_lon, min_lat, cur_max_lon, max_lat)
            sample = GoldenSample(
                sample_id=sid,
                dataset_name="golden-bitemporal-change-v3",
                source="Sentinel-2 MSI Pair",
                license="CDSE Open Access",
                acquisition_time_utc=cur_time,
                sensor=SensorType.S2_MSI,
                crs="EPSG:4326",
                gsd_m=10.0,
                bounds_wgs84=bounds_geom,
                ground_truth=(target,),
                checksum=_compute_sha256((arr_t0, arr_t1)),
                split=split,
                metadata={"t0_time": (cur_time - timedelta(days=10)).isoformat(), "t1_time": cur_time.isoformat()},
                image_data=(arr_t0, arr_t1),
                area_km2=float((h * 10.0 * w * 10.0) / 1_000_000.0),
            )
            split_samples[split].append(sample)

    dataset = GoldenDataset(
        name="golden-bitemporal-change-v3",
        train_samples=tuple(split_samples[DatasetSplit.TRAIN]),
        validation_samples=tuple(split_samples[DatasetSplit.VALIDATION]),
        test_samples=tuple(split_samples[DatasetSplit.TEST]),
        guard_window_seconds=86400.0 * 30.0,
    )
    dataset.validate()
    return dataset
