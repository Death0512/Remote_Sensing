import ast
from pathlib import Path
import pytest
import yaml

from src.provenance.builder import ProvenanceBuilder
from src.provenance.engine import (
    ModelArtifactValidator,
    ProvenanceEngine,
    compute_bytes_checksum,
    compute_config_hash,
    compute_file_checksum,
    get_git_commit,
)
from src.provenance.exceptions import (
    ModelChecksumMismatchError,
    ModelRegistryError,
    ModelUnavailableError,
)

FORBIDDEN_PLACEHOLDER_STRINGS = (
    "30d2bc8",
    "cli-cfg-v2",
    "cli-input-sha",
    "cli-output-sha",
    "live-bytes",
)


def _get_active_src_files() -> list[Path]:
    repo_root = Path(__file__).resolve().parent.parent.parent
    src_dir = repo_root / "src"
    frozen_legacy = frozenset({"core", "detectors", "engine"})
    active: list[Path] = []
    for py_file in src_dir.rglob("*.py"):
        rel_parts = py_file.relative_to(src_dir).parts
        if rel_parts and rel_parts[0] in frozen_legacy:
            continue
        active.append(py_file)
    return active


def test_ast_guard_no_placeholders_in_active_tree() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    files = _get_active_src_files()
    assert len(files) > 0

    violations: list[tuple[str, int, str]] = []
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                val = node.value
                for forbidden in FORBIDDEN_PLACEHOLDER_STRINGS:
                    if forbidden in val:
                        violations.append(
                            (str(f.relative_to(repo_root)), getattr(node, "lineno", 0), val)
                        )

    assert violations == [], f"Detected forbidden placeholder strings: {violations}"


def test_compute_bytes_and_file_checksum(tmp_path: Path) -> None:
    data = b"remote-sensing-test-data"
    expected = "5ee75320ab95068b199e26afdc6aa777ccd79cb038e810309a3ac60ad496715d"
    assert compute_bytes_checksum(data) == expected

    fpath = tmp_path / "sample.bin"
    fpath.write_bytes(data)
    assert compute_file_checksum(fpath) == expected


def test_compute_config_hash_dict_and_file(tmp_path: Path) -> None:
    cfg = {"sensor": "S1_SAR", "threshold": 0.85, "mode": "scan"}
    h1 = compute_config_hash(cfg)
    assert len(h1) == 64

    fpath = tmp_path / "config.yaml"
    fpath.write_text("sensor: S1_SAR\nthreshold: 0.85\n", encoding="utf-8")
    h2 = compute_config_hash(fpath)
    assert len(h2) == 64
    assert h1 != h2


def test_get_git_commit_valid() -> None:
    commit = get_git_commit()
    assert isinstance(commit, str)
    assert len(commit) >= 7
    assert commit != "unknown"
    assert commit != "30d2bc8"


def test_provenance_builder_runtime(tmp_path: Path) -> None:
    inp = tmp_path / "in.tif"
    inp.write_bytes(b"INPUT_BYTES_V3")
    out = tmp_path / "out.tif"
    out.write_bytes(b"OUTPUT_BYTES_V3")

    builder = ProvenanceBuilder(
        pipeline_version="3.0.0",
        config_source={"detector": "cfar", "pfa": 1e-6},
    )
    prov = builder.build(
        source_asset_id="asset-test-01",
        acquisition_metadata_ref="meta-test-01",
        input_data=inp,
        output_data=out,
        model_id="sar_vessel_cfar",
    )

    assert prov.schema_version == "provenance.v3"
    assert prov.git_commit != "30d2bc8"
    assert prov.input_checksum == compute_file_checksum(inp)
    assert prov.output_checksum == compute_file_checksum(out)
    assert prov.model_id == "sar_vessel_cfar"


def test_model_artifact_validator_pending_raises(tmp_path: Path) -> None:
    registry_yaml = tmp_path / "model_registry.yaml"
    registry_yaml.write_text(
        yaml.dump(
            {
                "models": {
                    "pending_model": {
                        "status": "PENDING",
                        "sha256": None,
                        "required": False,
                        "local_path": "data/weights/pending.pt",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    validator = ModelArtifactValidator(registry_path=registry_yaml)
    with pytest.raises(ModelUnavailableError, match="MODEL_UNAVAILABLE"):
        validator.validate_and_resolve("pending_model")


def test_model_artifact_validator_missing_raises(tmp_path: Path) -> None:
    registry_yaml = tmp_path / "model_registry.yaml"
    registry_yaml.write_text(
        yaml.dump(
            {
                "models": {
                    "yolo_test": {
                        "status": "AVAILABLE",
                        "sha256": "abcdef0123456789",
                        "required": True,
                        "local_path": str(tmp_path / "non_existent.pt"),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    validator = ModelArtifactValidator(registry_path=registry_yaml)
    with pytest.raises(ModelUnavailableError, match="MODEL_UNAVAILABLE"):
        validator.validate_and_resolve("yolo_test")


def test_model_artifact_validator_checksum_mismatch(tmp_path: Path) -> None:
    weight_file = tmp_path / "corrupted.pt"
    weight_file.write_bytes(b"CORRUPTED_WEIGHTS")

    registry_yaml = tmp_path / "model_registry.yaml"
    registry_yaml.write_text(
        yaml.dump(
            {
                "models": {
                    "corrupted_model": {
                        "status": "AVAILABLE",
                        "sha256": "1111222233334444555566667777888899990000111122223333444455556666",
                        "required": True,
                        "local_path": str(weight_file),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    validator = ModelArtifactValidator(registry_path=registry_yaml)
    with pytest.raises(ModelChecksumMismatchError, match="MODEL_CHECKSUM_MISMATCH"):
        validator.validate_and_resolve("corrupted_model")


def test_model_artifact_validator_success(tmp_path: Path) -> None:
    weight_file = tmp_path / "valid.pt"
    content = b"VALID_MODEL_WEIGHTS"
    weight_file.write_bytes(content)
    real_sha = compute_bytes_checksum(content)

    registry_yaml = tmp_path / "model_registry.yaml"
    registry_yaml.write_text(
        yaml.dump(
            {
                "models": {
                    "valid_model": {
                        "status": "AVAILABLE",
                        "sha256": real_sha,
                        "required": True,
                        "local_path": str(weight_file),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    validator = ModelArtifactValidator(registry_path=registry_yaml)
    resolved = validator.validate_and_resolve("valid_model")
    assert resolved == weight_file.resolve()
