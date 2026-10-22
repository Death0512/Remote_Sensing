from datetime import datetime, timezone
import hashlib
import math
from typing import List, Optional, Tuple
from uuid import uuid4
import numpy as np
import scipy.ndimage

from src.domain.enums import Capability, EvidenceKind
from src.domain.evidence import Evidence
from src.domain.observation import ProcessedObservation
from src.domain.provenance import Provenance


class ParallaxInflightDetector:
    capability: Capability = Capability.HEURISTIC

    def __init__(
        self,
        pixel_spacing_m: float = 10.0,
        inter_band_delay_s: float = 0.52,
        min_displacement_px: float = 2.0,
        max_displacement_px: float = 30.0,
        triplet_colinearity_max_deg: float = 25.0,
        min_velocity_kmh: float = 250.0,
        max_velocity_kmh: float = 1200.0,
        peak_threshold: float = 0.7,
    ) -> None:
        self.pixel_spacing_m = pixel_spacing_m
        self.inter_band_delay_s = inter_band_delay_s
        self.min_displacement_px = min_displacement_px
        self.max_displacement_px = max_displacement_px
        self.triplet_colinearity_max_deg = triplet_colinearity_max_deg
        self.min_velocity_kmh = min_velocity_kmh
        self.max_velocity_kmh = max_velocity_kmh
        self.peak_threshold = peak_threshold

    def _extract_peaks(
        self,
        band: np.ndarray,
        threshold: float,
    ) -> List[Tuple[float, float, float]]:
        mask = band >= threshold
        labeled, num_features = scipy.ndimage.label(mask)
        if num_features == 0:
            return []

        centers = scipy.ndimage.center_of_mass(
            mask, labeled, range(1, num_features + 1)
        )
        peaks: List[Tuple[float, float, float]] = []
        for r, c in centers:
            val = float(band[int(round(r)), int(round(c))])
            peaks.append((float(c), float(r), val))
        return peaks

    def _angle_between_vectors(
        self,
        v1: Tuple[float, float],
        v2: Tuple[float, float],
    ) -> float:
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        norm1 = math.hypot(v1[0], v1[1])
        norm2 = math.hypot(v2[0], v2[1])
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 180.0
        cos_theta = max(-1.0, min(1.0, dot / (norm1 * norm2)))
        return math.degrees(math.acos(cos_theta))

    def detect(
        self,
        processed_observation: ProcessedObservation,
        image_rgb: np.ndarray,
        pixel_spacing_m: Optional[float] = None,
        inter_band_delay_s: Optional[float] = None,
        peak_threshold: Optional[float] = None,
    ) -> Tuple[Evidence, ...]:
        spacing = (
            pixel_spacing_m
            if pixel_spacing_m is not None
            else self.pixel_spacing_m
        )
        delay_s = (
            inter_band_delay_s
            if inter_band_delay_s is not None
            else self.inter_band_delay_s
        )
        threshold = (
            peak_threshold
            if peak_threshold is not None
            else self.peak_threshold
        )

        if image_rgb.ndim != 3:
            raise ValueError(
                f"Expected 3-channel optical image, got shape {image_rgb.shape}"
            )

        if image_rgb.shape[0] == 3 and image_rgb.shape[0] < image_rgb.shape[1]:
            r_band = image_rgb[0].astype(np.float32)
            g_band = image_rgb[1].astype(np.float32)
            b_band = image_rgb[2].astype(np.float32)
        elif image_rgb.shape[2] == 3:
            r_band = image_rgb[:, :, 0].astype(np.float32)
            g_band = image_rgb[:, :, 1].astype(np.float32)
            b_band = image_rgb[:, :, 2].astype(np.float32)
        else:
            raise ValueError(
                f"Unsupported 3-channel layout for shape {image_rgb.shape}"
            )

        r_peaks = self._extract_peaks(r_band, threshold)
        g_peaks = self._extract_peaks(g_band, threshold)
        b_peaks = self._extract_peaks(b_band, threshold)

        evidences: List[Evidence] = []
        for rc, rr, _ in r_peaks:
            for gc, gr, _ in g_peaks:
                disp_rg = math.hypot(gc - rc, gr - rr)
                if not (
                    self.min_displacement_px
                    <= disp_rg
                    <= self.max_displacement_px
                ):
                    continue
                v_rg = (gc - rc, gr - rr)

                for bc, br, _ in b_peaks:
                    disp_gb = math.hypot(bc - gc, br - gr)
                    if not (
                        self.min_displacement_px
                        <= disp_gb
                        <= self.max_displacement_px
                    ):
                        continue
                    v_gb = (bc - gc, br - gr)

                    angle = self._angle_between_vectors(v_rg, v_gb)
                    if angle > self.triplet_colinearity_max_deg:
                        continue

                    total_disp_px = math.hypot(bc - rc, br - rr)
                    total_disp_m = total_disp_px * spacing
                    speed_mps = total_disp_m / max(delay_s, 1e-6)
                    speed_kmh = speed_mps * 3.6
                    if not (
                        self.min_velocity_kmh
                        <= speed_kmh
                        <= self.max_velocity_kmh
                    ):
                        continue

                    quality = max(
                        0.1,
                        min(
                            1.0,
                            1.0
                            - (
                                angle
                                / max(self.triplet_colinearity_max_deg, 1.0)
                            ),
                        ),
                    )
                    uncertainty_kmh = (
                        (0.5 * spacing / max(delay_s, 1e-6)) * 3.6
                    )

                    prov = Provenance(
                        provenance_id=f"prov-ev-kin-{uuid4().hex[:12]}",
                        source_asset_id=processed_observation.provenance.source_asset_id,
                        acquisition_metadata_ref=processed_observation.provenance.acquisition_metadata_ref,
                        pipeline_version="2.0.0",
                        git_commit=processed_observation.provenance.git_commit,
                        config_hash=processed_observation.provenance.config_hash,
                        model_id=None,
                        model_version=None,
                        input_checksum=processed_observation.provenance.output_checksum,
                        output_checksum=hashlib.sha256(
                            f"{rc}_{rr}_{speed_kmh:.1f}".encode("utf-8")
                        ).hexdigest()[:16],
                        processing_run_id=processed_observation.provenance.processing_run_id,
                        processing_time_utc=datetime.now(timezone.utc),
                        parent_provenance_id=processed_observation.provenance.provenance_id,
                    )

                    ev = Evidence(
                        evidence_id=f"ev-kin-{uuid4().hex[:12]}",
                        kind=EvidenceKind.KINEMATICS,
                        value=float(speed_kmh),
                        quality=float(quality),
                        source="ParallaxInflightDetector",
                        timestamp_utc=processed_observation.provenance.processing_time_utc,
                        uncertainty=float(uncertainty_kmh),
                        supports="EXPERIMENTAL_INFLIGHT_TARGET",
                        references=(processed_observation.processed_id,),
                        provenance=prov,
                    )
                    evidences.append(ev)

        return tuple(evidences)
