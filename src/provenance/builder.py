from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union
from uuid import uuid4

from src.domain.provenance import Provenance
from src.provenance.engine import (
    ProvenanceEngine,
    compute_bytes_checksum,
    compute_config_hash,
    compute_file_checksum,
    get_git_commit,
)


class ProvenanceBuilder:
    def __init__(
        self,
        pipeline_version: str = "3.0.0",
        config_source: Optional[Union[str, Path, bytes, Mapping[str, Any], Sequence[Any]]] = None,
        repo_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.pipeline_version = pipeline_version
        self.git_commit = get_git_commit(repo_path)
        if config_source is not None:
            self.config_hash = compute_config_hash(config_source)
        else:
            self.config_hash = compute_bytes_checksum(f"pipeline-{pipeline_version}".encode("utf-8"))

    def build(
        self,
        source_asset_id: str,
        acquisition_metadata_ref: str,
        input_data: Union[bytes, bytearray, memoryview, str, Path],
        output_data: Union[bytes, bytearray, memoryview, str, Path],
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        processing_run_id: Optional[str] = None,
        parent_provenance_id: Optional[str] = None,
        provenance_id: Optional[str] = None,
        processing_time_utc: Optional[datetime] = None,
    ) -> Provenance:
        in_checksum = (
            compute_file_checksum(input_data)
            if isinstance(input_data, (str, Path)) and Path(input_data).is_file()
            else compute_bytes_checksum(
                input_data if isinstance(input_data, (bytes, bytearray, memoryview)) else str(input_data).encode("utf-8")
            )
        )
        out_checksum = (
            compute_file_checksum(output_data)
            if isinstance(output_data, (str, Path)) and Path(output_data).is_file()
            else compute_bytes_checksum(
                output_data if isinstance(output_data, (bytes, bytearray, memoryview)) else str(output_data).encode("utf-8")
            )
        )

        return Provenance(
            provenance_id=provenance_id or f"prov-{uuid4().hex[:12]}",
            source_asset_id=source_asset_id,
            acquisition_metadata_ref=acquisition_metadata_ref,
            pipeline_version=self.pipeline_version,
            git_commit=self.git_commit,
            config_hash=self.config_hash,
            model_id=model_id,
            model_version=model_version,
            input_checksum=in_checksum,
            output_checksum=out_checksum,
            processing_run_id=processing_run_id or f"run-{uuid4().hex[:8]}",
            processing_time_utc=processing_time_utc or datetime.now(timezone.utc),
            parent_provenance_id=parent_provenance_id,
        )
