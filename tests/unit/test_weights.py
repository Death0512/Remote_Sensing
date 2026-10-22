import hashlib
import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.ingestion.weights import WeightDownloader


@pytest.fixture
def mock_registry(tmp_path: Path) -> Path:
    reg_file = tmp_path / "model_registry.yaml"
    content = """
models:
  test_model:
    task: "test_task"
    architecture: "TestArch"
    url: "https://example.com/test_model.pt"
    sha256: "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    local_path: "data/weights/test_model.pt"
  bad_sha_model:
    task: "test_task"
    architecture: "TestArch"
    url: "https://example.com/bad_sha.pt"
    sha256: "0000000000000000000000000000000000000000000000000000000000000000"
    local_path: "data/weights/bad_sha.pt"
"""
    reg_file.write_text(content.strip(), encoding="utf-8")
    return reg_file


def test_ensure_weight_already_exists(mock_registry: Path, tmp_path: Path):
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir(parents=True)
    existing_file = weights_dir / "test_model.pt"
    existing_file.write_bytes(b"existing content")

    downloader = WeightDownloader(registry_path=str(mock_registry), weights_dir=str(weights_dir))
    result = downloader.ensure_weight("test_model")
    assert result == existing_file
    assert result.read_bytes() == b"existing content"


def test_ensure_weight_download_and_sha_success(mock_registry: Path, tmp_path: Path):
    weights_dir = tmp_path / "weights"
    payload = b"test"
    computed_sha = hashlib.sha256(payload).hexdigest()
    assert computed_sha == "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"

    mock_resp = MagicMock()
    mock_resp.iter_content.return_value = [payload]
    mock_resp.raise_for_status = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("requests.get", return_value=mock_resp):
        downloader = WeightDownloader(registry_path=str(mock_registry), weights_dir=str(weights_dir))
        target_path = downloader.ensure_weight("test_model")

    assert target_path.exists()
    assert target_path.read_bytes() == payload
    part_file = target_path.with_suffix(target_path.suffix + ".part")
    assert not part_file.exists()


def test_ensure_weight_sha_mismatch(mock_registry: Path, tmp_path: Path):
    weights_dir = tmp_path / "weights"
    payload = b"test"

    mock_resp = MagicMock()
    mock_resp.iter_content.return_value = [payload]
    mock_resp.raise_for_status = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("requests.get", return_value=mock_resp):
        downloader = WeightDownloader(registry_path=str(mock_registry), weights_dir=str(weights_dir))
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            downloader.ensure_weight("bad_sha_model", max_retries=1)

    target_path = weights_dir / "bad_sha.pt"
    assert not target_path.exists()
    assert not target_path.with_suffix(target_path.suffix + ".part").exists()


def test_ensure_weight_retry_backoff(mock_registry: Path, tmp_path: Path):
    weights_dir = tmp_path / "weights"
    payload = b"test"

    mock_resp_fail = MagicMock()
    mock_resp_fail.raise_for_status.side_effect = RuntimeError("network glitch")
    mock_resp_fail.__enter__.return_value = mock_resp_fail
    mock_resp_fail.__exit__.return_value = None

    mock_resp_ok = MagicMock()
    mock_resp_ok.iter_content.return_value = [payload]
    mock_resp_ok.raise_for_status = MagicMock()
    mock_resp_ok.__enter__.return_value = mock_resp_ok
    mock_resp_ok.__exit__.return_value = None

    downloader = WeightDownloader(registry_path=str(mock_registry), weights_dir=str(weights_dir))
    with patch("requests.get", side_effect=[mock_resp_fail, mock_resp_ok]), \
         patch("time.sleep", return_value=None) as mock_sleep:
        target_path = downloader.ensure_weight("test_model", max_retries=2, backoff_factor=0.1)

    assert target_path.exists()
    assert target_path.read_bytes() == payload
    assert mock_sleep.call_count == 1
