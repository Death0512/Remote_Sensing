from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import requests
from shapely.geometry import box

from src.api.auth import Role, create_access_token
from src.api.server import app, _fetch_tile
from fastapi.testclient import TestClient
from src.domain.enums import Capability, SensorType
from src.domain.observation import Observation
from src.domain.provenance import Provenance
from src.ingestion.download.windowed_downloader import WindowedDownloader
from src.ingestion.exceptions import (
    DataCorruptedError,
    DataUnavailableError,
    InvalidGeoreferenceError,
    InvalidMetadataError,
)
from src.ingestion.providers.mock_provider import MockSatelliteProvider
from src.ingestion.validation.metadata_validator import MetadataValidator


def test_mock_provider_fatal_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    with pytest.raises(RuntimeError, match="MockSatelliteProvider forbidden in production/development environment"):
        MockSatelliteProvider()


def test_mock_provider_fatal_in_development(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "development")
    with pytest.raises(RuntimeError, match="MockSatelliteProvider forbidden in production/development environment"):
        MockSatelliteProvider()


def test_mock_provider_explicit_env_param() -> None:
    with pytest.raises(RuntimeError, match="MockSatelliteProvider forbidden"):
        MockSatelliteProvider(app_env="production")


def test_metadata_validator_crs_valid() -> None:
    crs_4326 = MetadataValidator.validate_crs("EPSG:4326")
    assert crs_4326.to_epsg() == 4326

    crs_3857 = MetadataValidator.validate_crs("EPSG:3857")
    assert crs_3857.to_epsg() == 3857


def test_metadata_validator_crs_invalid() -> None:
    with pytest.raises(InvalidGeoreferenceError, match="cannot be resolved"):
        MetadataValidator.validate_crs("INVALID_CRS_NOT_EXISTING")

    with pytest.raises(InvalidGeoreferenceError, match="required and cannot be empty"):
        MetadataValidator.validate_crs("")

    with pytest.raises(InvalidGeoreferenceError, match="required and cannot be empty"):
        MetadataValidator.validate_crs(None)


def test_metadata_validator_affine_valid() -> None:
    transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0)
    res = MetadataValidator.validate_affine_transform(transform)
    assert res == transform


def test_metadata_validator_affine_missing() -> None:
    with pytest.raises(InvalidGeoreferenceError, match="required and cannot be None"):
        MetadataValidator.validate_affine_transform(None)


def test_metadata_validator_affine_wrong_length() -> None:
    with pytest.raises(InvalidGeoreferenceError, match="must have 6 coefficients"):
        MetadataValidator.validate_affine_transform((1.0, 2.0, 3.0))


def test_metadata_validator_affine_degenerate() -> None:
    with pytest.raises(InvalidGeoreferenceError, match="degenerate with determinant"):
        MetadataValidator.validate_affine_transform((0.0, 0.0, 100.0, 0.0, 0.0, 200.0))

    with pytest.raises(InvalidGeoreferenceError, match="degenerate with determinant"):
        MetadataValidator.validate_affine_transform((2.0, 4.0, 100.0, 1.0, 2.0, 200.0))


def test_metadata_validator_affine_identity_rejected() -> None:
    with pytest.raises(InvalidGeoreferenceError, match="unreferenced identity matrix"):
        MetadataValidator.validate_affine_transform((1.0, 0.0, 0.0, 0.0, 1.0, 0.0))

    with pytest.raises(InvalidGeoreferenceError, match="unreferenced identity matrix"):
        MetadataValidator.validate_affine_transform((1.0, 0.0, 0.0, 0.0, -1.0, 0.0))


def test_metadata_validator_gsd_consistency_projected() -> None:
    crs_32649 = MetadataValidator.validate_crs("EPSG:32649")
    transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0)
    gsd = MetadataValidator.validate_gsd(10.0, transform, crs_32649)
    assert gsd == 10.0


def test_metadata_validator_gsd_consistency_geographic() -> None:
    crs_4326 = MetadataValidator.validate_crs("EPSG:4326")
    transform = (0.00009, 0.0, 108.0, 0.0, -0.00009, 16.0)
    gsd = MetadataValidator.validate_gsd(10.0, transform, crs_4326)
    assert gsd == 10.0


def test_metadata_validator_gsd_inconsistent_raises() -> None:
    crs_32649 = MetadataValidator.validate_crs("EPSG:32649")
    transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0)
    with pytest.raises(InvalidMetadataError, match="inconsistent with affine scale"):
        MetadataValidator.validate_gsd(100.0, transform, crs_32649)


def test_metadata_validator_gsd_non_positive_raises() -> None:
    crs_32649 = MetadataValidator.validate_crs("EPSG:32649")
    transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0)
    with pytest.raises(InvalidMetadataError, match="must be positive"):
        MetadataValidator.validate_gsd(-5.0, transform, crs_32649)


def test_metadata_validator_acquisition_datetimes() -> None:
    now_utc = datetime.now(timezone.utc)
    later_utc = now_utc + timedelta(hours=1)
    s, e = MetadataValidator.validate_acquisitions(now_utc, later_utc)
    assert s == now_utc
    assert e == later_utc

    iso_s = "2026-10-22T10:00:00Z"
    iso_e = "2026-10-22T10:05:00+00:00"
    s2, e2 = MetadataValidator.validate_acquisitions(iso_s, iso_e)
    assert s2.tzinfo == timezone.utc
    assert e2.tzinfo == timezone.utc


def test_metadata_validator_acquisition_naive_rejected() -> None:
    naive_dt = datetime(2026, 10, 22, 10, 0, 0)
    with pytest.raises(InvalidMetadataError, match="must be a timezone-aware UTC datetime"):
        MetadataValidator.validate_acquisitions(naive_dt)


def test_metadata_validator_acquisition_end_before_start() -> None:
    now_utc = datetime.now(timezone.utc)
    earlier = now_utc - timedelta(hours=1)
    with pytest.raises(InvalidMetadataError, match="cannot be earlier than start"):
        MetadataValidator.validate_acquisitions(now_utc, earlier)


def test_metadata_validator_source_sensor_product_match() -> None:
    src, sens, prod = MetadataValidator.validate_source_sensor_product("cdse", "S1_SAR", "GRD")
    assert src == "cdse"
    assert sens == SensorType.S1_SAR
    assert prod == "GRD"

    src, sens, prod = MetadataValidator.validate_source_sensor_product(
        "planetary_computer", SensorType.S2_MSI, "L2A"
    )
    assert src == "planetary_computer"
    assert sens == SensorType.S2_MSI
    assert prod == "L2A"

    src, sens, prod = MetadataValidator.validate_source_sensor_product("vhr", "VHR_OPTICAL", "RGB")
    assert src == "vhr"
    assert sens == SensorType.VHR_OPTICAL
    assert prod == "RGB"


def test_metadata_validator_mismatch_raises() -> None:
    with pytest.raises(InvalidMetadataError, match="not valid for declared source"):
        MetadataValidator.validate_source_sensor_product("cdse", "VHR_OPTICAL", "RGB")

    with pytest.raises(InvalidMetadataError, match="does not match source"):
        MetadataValidator.validate_source_sensor_product("cdse", "S1_SAR", "L2A")

    with pytest.raises(InvalidMetadataError, match="Unsupported declared source"):
        MetadataValidator.validate_source_sensor_product("unknown_sat_provider", "S1_SAR", "GRD")


def test_metadata_validator_mock_source_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    with pytest.raises(InvalidMetadataError, match="Mock source forbidden"):
        MetadataValidator.validate_source_sensor_product("mock", "S1_SAR", "GRD")


def test_metadata_validator_validate_full_dict() -> None:
    payload = {
        "source": "cdse",
        "sensor": "S1_SAR",
        "product": "GRD",
        "crs": "EPSG:32649",
        "ground_sample_distance_m": 10.0,
        "acquisition_start": "2026-10-22T04:00:00Z",
        "acquisition_end": "2026-10-22T04:01:00Z",
        "affine_transform": (10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0),
    }
    result = MetadataValidator.validate(payload)
    assert result["is_valid"] is True
    assert result["source"] == "cdse"
    assert result["sensor"] == SensorType.S1_SAR


def test_metadata_validator_validate_observation_object() -> None:
    now = datetime.now(timezone.utc)
    prov = Provenance(
        provenance_id="prov-1",
        source_asset_id="asset-1",
        acquisition_metadata_ref="meta://1",
        pipeline_version="3.0.0",
        git_commit="HEAD",
        config_hash="cfg-hash",
        model_id=None,
        model_version=None,
        input_checksum="input-sha",
        output_checksum="output-sha",
        processing_run_id="run-1",
        processing_time_utc=now,
    )
    obs = Observation(
        observation_id="obs-1",
        source="cdse",
        platform="Sentinel-1A",
        sensor=SensorType.S1_SAR,
        product="GRD",
        processing_level="LEVEL1",
        acquisition_time_utc=now,
        crs="EPSG:32649",
        ground_sample_distance_m=10.0,
        bounds_wgs84=box(108.0, 16.0, 109.0, 17.0),
        asset_uri="file:///tmp/asset.tif",
        metadata_uri="file:///tmp/meta.json",
        quality_flags=(),
        provenance=prov,
    )
    transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0)
    result = MetadataValidator.validate(obs, affine_transform=transform)
    assert result["is_valid"] is True
    assert result["source"] == "cdse"


def test_metadata_validator_file_missing() -> None:
    with pytest.raises(DataUnavailableError, match="not found"):
        MetadataValidator.validate_file("/tmp/non_existent_metadata_file_xyz.json")


def test_metadata_validator_file_corrupt(tmp_path: Path) -> None:
    corrupt_file = tmp_path / "corrupt_meta.json"
    corrupt_file.write_text("NOT_VALID_JSON{:::}")
    with pytest.raises(DataCorruptedError, match="corrupted or invalid JSON"):
        MetadataValidator.validate_file(corrupt_file)


def test_metadata_validator_file_checksum_mismatch(tmp_path: Path) -> None:
    meta_file = tmp_path / "valid_meta.json"
    payload = {
        "source": "cdse",
        "sensor": "S1_SAR",
        "product": "GRD",
        "crs": "EPSG:32649",
        "ground_sample_distance_m": 10.0,
        "acquisition_start": "2026-10-22T04:00:00Z",
        "affine_transform": (10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0),
    }
    meta_file.write_text(json.dumps(payload))
    with pytest.raises(DataCorruptedError, match="checksum mismatch"):
        MetadataValidator.validate_file(meta_file, expected_sha256="0000000000000000000000000000000000000000000000000000000000000000")


def test_metadata_validator_file_valid(tmp_path: Path) -> None:
    meta_file = tmp_path / "valid_meta.json"
    payload = {
        "source": "cdse",
        "sensor": "S1_SAR",
        "product": "GRD",
        "crs": "EPSG:32649",
        "ground_sample_distance_m": 10.0,
        "acquisition_start": "2026-10-22T04:00:00Z",
        "affine_transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0],
    }
    content = json.dumps(payload).encode("utf-8")
    meta_file.write_bytes(content)
    expected_sha = hashlib.sha256(content).hexdigest()

    result = MetadataValidator.validate_file(meta_file, expected_sha256=expected_sha)
    assert result["is_valid"] is True
    assert result["source"] == "cdse"


def test_windowed_downloader_compute_and_verify_sha(tmp_path: Path) -> None:
    sample_file = tmp_path / "data.bin"
    sample_data = b"RECONNAISSANCE_STREAM_DATA_TEST"
    sample_file.write_bytes(sample_data)
    expected_sha = hashlib.sha256(sample_data).hexdigest()

    computed_sha = WindowedDownloader.compute_sha256(sample_file)
    assert computed_sha == expected_sha
    assert WindowedDownloader.verify_checksum(sample_file, expected_sha) is True
    assert WindowedDownloader.verify_checksum(sample_file, "bad_hash") is False


def test_windowed_downloader_local_window_slice(tmp_path: Path) -> None:
    sample_file = tmp_path / "window_source.bin"
    sample_file.write_bytes(b"ABCDEFGHIJKLMN")

    downloader = WindowedDownloader()
    window = downloader.download_window(sample_file, start_byte=3, end_byte=6)
    assert window == b"DEFG"

    expected_window_sha = hashlib.sha256(b"DEFG").hexdigest()
    out_file = tmp_path / "window_out.bin"
    res_path = downloader.download_window(
        sample_file,
        start_byte=3,
        end_byte=6,
        output_path=out_file,
        expected_sha256=expected_window_sha,
    )
    assert res_path == out_file
    assert out_file.read_bytes() == b"DEFG"


def test_windowed_downloader_corrupt_window_checksum(tmp_path: Path) -> None:
    sample_file = tmp_path / "window_source.bin"
    sample_file.write_bytes(b"0123456789")

    downloader = WindowedDownloader()
    with pytest.raises(DataCorruptedError, match="SHA-256 mismatch for window"):
        downloader.download_window(
            sample_file,
            start_byte=0,
            end_byte=4,
            expected_sha256="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        )


def test_windowed_downloader_file_download_success(tmp_path: Path) -> None:
    src_file = tmp_path / "full_src.bin"
    payload = b"STREAMING_PAYLOAD_" * 1024
    src_file.write_bytes(payload)
    expected_sha = hashlib.sha256(payload).hexdigest()

    dst_file = tmp_path / "full_dst.bin"
    downloader = WindowedDownloader(chunk_size=512)
    res = downloader.download_file(src_file, dst_file, expected_sha256=expected_sha)
    assert res == dst_file
    assert dst_file.read_bytes() == payload
    assert not dst_file.with_name(f"{dst_file.name}.part").exists()


def test_windowed_downloader_file_download_corrupt_cleans_part(tmp_path: Path) -> None:
    src_file = tmp_path / "full_src.bin"
    payload = b"CORRUPT_PAYLOAD"
    src_file.write_bytes(payload)

    dst_file = tmp_path / "full_dst.bin"
    downloader = WindowedDownloader()
    with pytest.raises(DataCorruptedError, match="Checksum mismatch"):
        downloader.download_file(src_file, dst_file, expected_sha256="badhash123")

    assert not dst_file.exists()
    assert not dst_file.with_name(f"{dst_file.name}.part").exists()


def test_windowed_downloader_missing_source() -> None:
    downloader = WindowedDownloader()
    with pytest.raises(DataUnavailableError, match="Local source file not found"):
        downloader.download_file("/tmp/non_existent_source_123.bin", "/tmp/out.bin")


def test_windowed_downloader_http_window_success() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 206
    mock_resp.content = b"HTTP_RANGE_DATA"

    mock_session = MagicMock()
    mock_session.get.return_value = mock_resp

    downloader = WindowedDownloader(session=mock_session)
    data = downloader.download_window(
        "https://imagery.dataspace.copernicus.eu/tile.tif",
        start_byte=0,
        end_byte=14,
    )
    assert data == b"HTTP_RANGE_DATA"
    mock_session.get.assert_called_once_with(
        "https://imagery.dataspace.copernicus.eu/tile.tif",
        headers={"Range": "bytes=0-14"},
        timeout=30.0,
    )


def test_windowed_downloader_http_404_raises_data_unavailable() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 404

    mock_session = MagicMock()
    mock_session.get.return_value = mock_resp

    downloader = WindowedDownloader(session=mock_session)
    with pytest.raises(DataUnavailableError, match="Resource unavailable"):
        downloader.download_window(
            "https://imagery.dataspace.copernicus.eu/missing.tif",
            start_byte=0,
            end_byte=10,
        )


def test_api_fetch_tile_raises_data_unavailable_on_failure() -> None:
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        with pytest.raises(DataUnavailableError, match="DATA_UNAVAILABLE: tile fetch failed"):
            _fetch_tile("satellite_hd", "2026-10-22", 14, 100, 200)


def test_api_scan_map_aoi_returns_503_on_tile_unavailable() -> None:
    client = TestClient(app)
    token = create_access_token(user_id="op-01", role=Role.OPERATOR)
    with patch("src.api.server._fetch_tile", side_effect=DataUnavailableError("DATA_UNAVAILABLE: imagery tile fetch failed")):
        resp = client.post(
            "/api/scan-map-aoi",
            json={"bbox": [108.0, 16.0, 108.1, 16.1]},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "DATA_UNAVAILABLE"
        assert any("DATA_UNAVAILABLE" in err["message"] for err in data.get("errors", []))
