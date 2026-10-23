import hashlib
from pathlib import Path
import subprocess
from typing import Any, Mapping, Optional, Sequence, Union
import yaml

from src.provenance.exceptions import (
    ModelChecksumMismatchError,
    ModelUnavailableError,
)


def get_git_commit(repo_path: Optional[Union[str, Path]] = None) -> str:
    cwd = Path(repo_path) if repo_path else Path(__file__).resolve().parent
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=5.0,
        )
        commit = proc.stdout.strip()
        if commit:
            return commit
    except Exception:
        pass
    return "unknown"


def compute_bytes_checksum(data: Union[bytes, bytearray, memoryview]) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_file_checksum(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for checksum: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_config_hash(
    config_source: Union[str, Path, bytes, Mapping[str, Any], Sequence[Any]]
) -> str:
    if isinstance(config_source, (str, Path)) and Path(config_source).is_file():
        return compute_file_checksum(config_source)
    if isinstance(config_source, (bytes, bytearray, memoryview)):
        return compute_bytes_checksum(config_source)
    if isinstance(config_source, str):
        return compute_bytes_checksum(config_source.encode("utf-8"))
    payload = yaml.safe_dump(config_source, sort_keys=True)
    return compute_bytes_checksum(payload.encode("utf-8"))


class ModelArtifactValidator:
    def __init__(
        self,
        registry_source: Optional[Union[str, Path, Mapping[str, Any]]] = None,
        registry_path: Optional[Union[str, Path]] = None,
    ) -> None:
        source = registry_source if registry_source is not None else registry_path
        if source is None:
            source = Path(__file__).resolve().parent.parent.parent / "configs" / "model_registry.yaml"

        if isinstance(source, (str, Path)):
            reg_path = Path(source)
            if not reg_path.is_file():
                raise ModelUnavailableError(f"Registry file not found: {reg_path}")
            with reg_path.open("r", encoding="utf-8") as f:
                parsed = yaml.safe_load(f) or {}
            self.registry: Mapping[str, Any] = parsed.get("models", parsed)
            self.registry_dir: Path = reg_path.parent
        elif isinstance(source, Mapping):
            self.registry = source.get("models", source)
            self.registry_dir = Path.cwd()
        else:
            raise TypeError(f"Unsupported registry source: {type(source)}")

    def validate_and_resolve(
        self,
        model_name: str,
        override_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        if model_name not in self.registry:
            raise ModelUnavailableError(f"MODEL_UNAVAILABLE: '{model_name}' not in registry")

        meta = self.registry[model_name]
        status = meta.get("status", "AVAILABLE")
        expected_sha = meta.get("sha256")
        configured_path_raw = override_path or meta.get("local_path")

        if not configured_path_raw:
            raise ModelUnavailableError(
                f"MODEL_UNAVAILABLE: Model '{model_name}' has no configured local_path"
            )

        candidate_path = Path(configured_path_raw)
        if not candidate_path.is_absolute():
            resolved = (Path.cwd() / candidate_path).resolve()
            if not resolved.is_file():
                resolved = (self.registry_dir.parent / candidate_path).resolve()
            candidate_path = resolved
        else:
            candidate_path = candidate_path.resolve()

        if status == "PENDING" and not candidate_path.is_file():
            raise ModelUnavailableError(
                f"MODEL_UNAVAILABLE: Model '{model_name}' is in PENDING status without weights"
            )

        if not candidate_path.is_file():
            raise ModelUnavailableError(
                f"MODEL_UNAVAILABLE: Checkpoint missing for '{model_name}' at {candidate_path}"
            )

        actual_sha = compute_file_checksum(candidate_path)

        if expected_sha:
            if actual_sha.lower() != expected_sha.lower():
                raise ModelChecksumMismatchError(
                    f"MODEL_CHECKSUM_MISMATCH: Model '{model_name}' expected {expected_sha}, got {actual_sha}"
                )
        elif status == "PENDING":
            raise ModelUnavailableError(
                f"MODEL_UNAVAILABLE: Model '{model_name}' has pending verification without digest"
            )

        return candidate_path


class ProvenanceEngine:
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        pipeline_version: str = "3.0.0",
        repo_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.pipeline_version = pipeline_version
        self.config_path = Path(config_path) if config_path else None
        self.repo_path = repo_path
        self._git_commit = get_git_commit(repo_path)
        self._config_hash = (
            compute_config_hash(self.config_path)
            if self.config_path and self.config_path.is_file()
            else compute_bytes_checksum(f"pipeline-{pipeline_version}".encode("utf-8"))
        )

    @property
    def git_commit(self) -> str:
        return self._git_commit

    @property
    def config_hash(self) -> str:
        return self._config_hash

    def compute_input_checksum(
        self, data: Union[bytes, bytearray, memoryview, str, Path]
    ) -> str:
        if isinstance(data, (bytes, bytearray, memoryview)):
            return compute_bytes_checksum(data)
        path = Path(data)
        if path.is_file():
            return compute_file_checksum(path)
        return compute_bytes_checksum(str(data).encode("utf-8"))

    def compute_output_checksum(
        self, data: Union[bytes, bytearray, memoryview, str, Path]
    ) -> str:
        if isinstance(data, (bytes, bytearray, memoryview)):
            return compute_bytes_checksum(data)
        path = Path(data)
        if path.is_file():
            return compute_file_checksum(path)
        return compute_bytes_checksum(str(data).encode("utf-8"))
