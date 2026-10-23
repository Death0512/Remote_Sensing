from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class Provenance:
    provenance_id: str
    source_asset_id: str
    acquisition_metadata_ref: str
    pipeline_version: str
    git_commit: str
    config_hash: str
    model_id: Optional[str]
    model_version: Optional[str]
    input_checksum: str
    output_checksum: str
    processing_run_id: str
    processing_time_utc: datetime
    parent_provenance_id: Optional[str] = None
    schema_version: str = "provenance.v3"
