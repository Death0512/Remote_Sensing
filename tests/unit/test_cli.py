import json
from pathlib import Path
import numpy as np

from src.cli.main import cli_entrypoint
from tests.mocks.mock_data import generate_synthetic_sar_scene, generate_synthetic_parallax_scene


def test_cli_help_and_empty():
    assert cli_entrypoint([]) == 1


def test_cli_status(capsys):
    ret = cli_entrypoint(["status"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "SURVEILLANCE STATUS" in captured.out
    assert "DATA SOURCES" in captured.out


def test_cli_generate_aoi(tmp_path: Path, capsys):
    aoi_dir = tmp_path / "aoi"
    ret = cli_entrypoint(["generate-aoi", "--output-dir", str(aoi_dir)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "AOI files generated" in captured.out
    assert (aoi_dir / "center_point.geojson").exists()
    assert (aoi_dir / "zone1_critical_500km.geojson").exists()
    assert (aoi_dir / "strategic_chokepoints.geojson").exists()


def test_cli_scan_sar(tmp_path: Path, capsys):
    sar_scene = generate_synthetic_sar_scene(
        height=100,
        width=100,
        targets=[(30, 30, 4, 4, 300.0)],
        seed=123,
    )
    img_path = tmp_path / "sar_scene.npy"
    np.save(str(img_path), sar_scene)

    out_dir = tmp_path / "outputs"
    ret = cli_entrypoint([
        "scan",
        "--image", str(img_path),
        "--sensor", "S1_GRD",
        "--output-dir", str(out_dir),
    ])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Scan complete" in captured.out

    alerts_file = out_dir / "alerts.geojson"
    assert alerts_file.exists()
    with open(alerts_file) as f:
        data = json.load(f)
    assert data["type"] == "FeatureCollection"
    assert "features" in data


def test_cli_scan_parallax(tmp_path: Path):
    scene = generate_synthetic_parallax_scene(height=80, width=80)
    img_path = tmp_path / "parallax_scene.npy"
    np.save(str(img_path), scene)

    out_dir = tmp_path / "out_opt"
    ret = cli_entrypoint([
        "scan",
        "--image", str(img_path),
        "--sensor", "S2_L1C",
        "--output-dir", str(out_dir),
    ])
    assert ret == 0
    alerts_file = out_dir / "alerts.geojson"
    assert alerts_file.exists()


def test_cli_scan_missing_image():
    ret = cli_entrypoint(["scan", "--image", "non_existent_file.npy"])
    assert ret == 1
