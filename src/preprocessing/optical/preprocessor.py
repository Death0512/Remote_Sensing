from datetime import datetime, timezone
import hashlib
from typing import Optional
from uuid import uuid4
import numpy as np

from src.domain.enums import Capability, GeometricState, QualityTier, RadiometricState
from src.domain.observation import Observation, ProcessedObservation
from src.domain.provenance import Provenance


class OpticalPreprocessor:
    capability: Capability = Capability.IMPLEMENTED

    def __init__(self, default_cloud_threshold: float = 0.30) -> None:
        self.default_cloud_threshold = default_cloud_threshold

    def normalize(
        self,
        data: np.ndarray,
        scale: float = 10000.0,
        offset: float = 0.0,
    ) -> np.ndarray:
        float_data = data.astype(np.float32)
        normalized = (float_data - offset) / max(scale, 1e-6)
        return np.clip(normalized, 0.0, 1.0)

    def compute_cloud_fraction(self, cloud_mask: np.ndarray) -> float:
        if cloud_mask.size == 0:
            return 0.0
        return float(np.mean(cloud_mask.astype(bool)))

    def process(
        self,
        observation: Observation,
        cloud_mask: Optional[np.ndarray] = None,
        cloud_threshold: Optional[float] = None,
    ) -> ProcessedObservation:
        threshold = (
            cloud_threshold
            if cloud_threshold is not None
            else self.default_cloud_threshold
        )

        cloud_fraction = 0.0
        if cloud_mask is not None:
            cloud_fraction = self.compute_cloud_fraction(cloud_mask)

        is_flagged_cloudy = "CLOUD_CONTAMINATED" in observation.quality_flags
        is_cloud_exceeded = cloud_fraction > threshold

        if is_flagged_cloudy or is_cloud_exceeded:
            quality_tier = QualityTier.CLOUD_CONTAMINATED
        elif "DATA_PARTIAL" in observation.quality_flags:
            quality_tier = QualityTier.DATA_PARTIAL
        elif "OUTSIDE_VALID_FOOTPRINT" in observation.quality_flags:
            quality_tier = QualityTier.OUTSIDE_VALID_FOOTPRINT
        else:
            quality_tier = QualityTier.DATA_OK

        if (
            "TOA_REFLECTANCE" in observation.quality_flags
            or observation.processing_level in {"L1C", "L2A", "TOA", "BOA"}
        ):
            radiometric_state = RadiometricState.TOA_REFLECTANCE
        else:
            radiometric_state = RadiometricState.RAW

        if observation.processing_level in {"L2A", "ORTHO"}:
            geometric_state = GeometricState.TERRAIN_CORRECTED
        else:
            geometric_state = GeometricState.GEOCODED

        masks = []
        if cloud_mask is not None:
            masks.append(("cloud_mask", f"cloud_fraction_{cloud_fraction:.3f}"))

        proc_prov = Provenance(
            provenance_id=f"prov-proc-opt-{uuid4().hex[:12]}",
            source_asset_id=observation.provenance.source_asset_id,
            acquisition_metadata_ref=observation.metadata_uri,
            pipeline_version="2.0.0",
            git_commit=observation.provenance.git_commit,
            config_hash=observation.provenance.config_hash,
            model_id=None,
            model_version=None,
            input_checksum=observation.provenance.output_checksum,
            output_checksum=hashlib.sha256(
                f"{observation.observation_id}_opt_proc".encode("utf-8")
            ).hexdigest()[:16],
            processing_run_id=f"run-{uuid4().hex[:8]}",
            processing_time_utc=datetime.now(timezone.utc),
            parent_provenance_id=observation.provenance.provenance_id,
        )

        return ProcessedObservation(
            processed_id=f"proc-{observation.observation_id}",
            observation_id=observation.observation_id,
            preprocessing_profile="optical_standard",
            radiometric_state=radiometric_state,
            geometric_state=geometric_state,
            masks=tuple(masks),
            quality_tier=quality_tier,
            provenance=proc_prov,
        )
