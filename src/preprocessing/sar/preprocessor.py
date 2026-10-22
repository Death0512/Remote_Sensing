from datetime import datetime, timezone
import hashlib
from typing import Optional
from uuid import uuid4
import numpy as np
from scipy.ndimage import binary_dilation

from src.domain.enums import Capability, GeometricState, QualityTier, RadiometricState
from src.domain.observation import Observation, ProcessedObservation
from src.domain.provenance import Provenance


class SarPreprocessor:
    capability: Capability = Capability.IMPLEMENTED

    def __init__(self, buffer_pixels: int = 2) -> None:
        self.buffer_pixels = max(0, buffer_pixels)

    def buffer_land_mask(self, land_mask: np.ndarray) -> np.ndarray:
        if self.buffer_pixels == 0:
            return land_mask.astype(bool)
        return binary_dilation(land_mask.astype(bool), iterations=self.buffer_pixels)

    def apply_land_mask(
        self,
        intensity: np.ndarray,
        land_mask: np.ndarray,
        fill_value: float = 0.0,
    ) -> np.ndarray:
        buffered = self.buffer_land_mask(land_mask)
        masked = np.copy(intensity)
        masked[buffered] = fill_value
        return masked

    def process(
        self,
        observation: Observation,
        land_mask: Optional[np.ndarray] = None,
    ) -> ProcessedObservation:
        has_calibration_meta = bool(
            observation.metadata_uri and observation.metadata_uri.strip()
        )
        is_calibrated_gamma0 = (
            "CALIBRATED_GAMMA0" in observation.quality_flags
            or observation.processing_level in {"RTC", "CALIBRATED_GAMMA0", "GAMMA0"}
        )
        is_calibrated_sigma0 = (
            "CALIBRATED_SIGMA0" in observation.quality_flags
            or observation.processing_level in {"CALIBRATED_SIGMA0", "SIGMA0"}
            or (observation.product == "GRD" and has_calibration_meta)
        )
        is_explicit_uncertain = (
            "RADIOMETRIC_UNCERTAIN" in observation.quality_flags
            or "UNCALIBRATED" in observation.quality_flags
        )

        if is_calibrated_gamma0 and has_calibration_meta and not is_explicit_uncertain:
            radiometric_state = RadiometricState.CALIBRATED_GAMMA0
            quality_tier = QualityTier.DATA_OK
        elif is_calibrated_sigma0 and has_calibration_meta and not is_explicit_uncertain:
            radiometric_state = RadiometricState.CALIBRATED_SIGMA0
            quality_tier = QualityTier.DATA_OK
        else:
            radiometric_state = RadiometricState.RAW
            quality_tier = QualityTier.RADIOMETRIC_UNCERTAIN

        if observation.processing_level == "RTC":
            geometric_state = GeometricState.TERRAIN_CORRECTED
        else:
            geometric_state = GeometricState.GEOCODED

        masks = []
        if land_mask is not None:
            masks.append(("land_mask", f"buffered_{self.buffer_pixels}px"))

        proc_prov = Provenance(
            provenance_id=f"prov-proc-{uuid4().hex[:12]}",
            source_asset_id=observation.provenance.source_asset_id,
            acquisition_metadata_ref=observation.metadata_uri,
            pipeline_version="2.0.0",
            git_commit=observation.provenance.git_commit,
            config_hash=observation.provenance.config_hash,
            model_id=None,
            model_version=None,
            input_checksum=observation.provenance.output_checksum,
            output_checksum=hashlib.sha256(
                f"{observation.observation_id}_proc".encode("utf-8")
            ).hexdigest()[:16],
            processing_run_id=f"run-{uuid4().hex[:8]}",
            processing_time_utc=datetime.now(timezone.utc),
            parent_provenance_id=observation.provenance.provenance_id,
        )

        return ProcessedObservation(
            processed_id=f"proc-{observation.observation_id}",
            observation_id=observation.observation_id,
            preprocessing_profile="sar_s1_standard",
            radiometric_state=radiometric_state,
            geometric_state=geometric_state,
            masks=tuple(masks),
            quality_tier=quality_tier,
            provenance=proc_prov,
        )
