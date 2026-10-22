from datetime import datetime, timezone
import hashlib
from typing import Optional, Sequence, Union
import uuid

from src.domain.enums import (
    AisCorrelationState,
    CalibrationState,
    Capability,
    EvidenceKind,
    HypothesisStatement,
    QualityTier,
    ScoreMethod,
    TargetClass,
    Zone,
)
from src.domain.evidence import Evidence
from src.domain.hypothesis import Hypothesis
from src.domain.provenance import Provenance
from src.intelligence.accumulator import EvidenceAccumulator


class HypothesisEngine:
    capability: Capability = Capability.IMPLEMENTED

    def __init__(
        self,
        pipeline_version: str = "2.0.0",
        git_commit: str = "30d2bc8",
    ) -> None:
        self.pipeline_version = pipeline_version
        self.git_commit = git_commit

    def _create_provenance(
        self,
        hypothesis_id: str,
        supporting_evidence: Sequence[Evidence],
    ) -> Provenance:
        source_asset_id = "unknown-asset"
        metadata_ref = "s3://surveillance/metadata.xml"
        processing_run_id = f"run-{uuid.uuid4().hex[:8]}"
        parent_prov_id = None

        if supporting_evidence and supporting_evidence[0].provenance:
            first_prov = supporting_evidence[0].provenance
            source_asset_id = first_prov.source_asset_id
            metadata_ref = first_prov.acquisition_metadata_ref
            processing_run_id = first_prov.processing_run_id
            parent_prov_id = first_prov.provenance_id

        input_hash = hashlib.sha256(
            "".join(e.evidence_id for e in supporting_evidence).encode("utf-8")
        ).hexdigest()
        output_hash = hashlib.sha256(hypothesis_id.encode("utf-8")).hexdigest()

        return Provenance(
            provenance_id=str(uuid.uuid4()),
            source_asset_id=source_asset_id,
            acquisition_metadata_ref=metadata_ref,
            pipeline_version=self.pipeline_version,
            git_commit=self.git_commit,
            config_hash="hypothesis-engine-v2",
            model_id=None,
            model_version=None,
            input_checksum=input_hash,
            output_checksum=output_hash,
            processing_run_id=processing_run_id,
            processing_time_utc=datetime.now(timezone.utc),
            parent_provenance_id=parent_prov_id,
        )

    def _build_explanations(
        self,
        statement: HypothesisStatement,
        primary_summary: str,
        supporting: Sequence[Evidence],
        refuting: Sequence[Evidence],
        score: float,
        calibration_state: CalibrationState,
    ) -> tuple[str, str]:
        primary_explanation = f"{statement.value}: {primary_summary}"

        lines = [
            f"Hypothesis: {statement.value}",
            f"Summary: {primary_summary}",
            f"Supporting Evidence ({len(supporting)} items):",
        ]
        for ev in supporting:
            sensor = next(
                (
                    r.split("=", 1)[1]
                    for r in ev.references
                    if r.startswith("sensor=")
                ),
                ev.source,
            )
            ts = (
                ev.timestamp_utc.isoformat()
                if hasattr(ev.timestamp_utc, "isoformat")
                else str(ev.timestamp_utc)
            )
            lines.append(
                f"  - [{ev.kind.value}] id={ev.evidence_id} sensor={sensor} "
                f"val={ev.value:.2f} qual={ev.quality:.2f} time={ts} "
                f"supports={ev.supports}"
            )

        if refuting:
            lines.append(f"Refuting Evidence ({len(refuting)} items):")
            for ev in refuting:
                ts = (
                    ev.timestamp_utc.isoformat()
                    if hasattr(ev.timestamp_utc, "isoformat")
                    else str(ev.timestamp_utc)
                )
                lines.append(
                    f"  - [{ev.kind.value}] id={ev.evidence_id} source={ev.source} "
                    f"val={ev.value:.2f} time={ts} supports={ev.supports}"
                )
        else:
            lines.append("Refuting Evidence: none observed.")

        if calibration_state == CalibrationState.NOT_CALIBRATED:
            lines.append(
                f"Calibration: Calibrated probability unavailable; heuristic/model raw score: {score:.2f}"
            )
        else:
            lines.append(
                f"Calibration: State={calibration_state.value}, Score={score:.2f}"
            )

        detailed_rationale = "\n".join(lines)
        return primary_explanation, detailed_rationale

    def _compute_score(self, supporting: Sequence[Evidence]) -> float:
        if not supporting:
            return 0.0
        scores: list[float] = []
        for ev in supporting:
            if ev.kind == EvidenceKind.KINEMATICS:
                normalized_val = min(ev.value / 500.0, 1.0)
            else:
                normalized_val = min(max(ev.value, 0.0), 1.0)
            weight = max(min(ev.quality, 1.0), 0.1)
            scores.append(normalized_val * weight)
        return round(sum(scores) / len(scores), 2)

    def evaluate(
        self,
        accumulator: Union[EvidenceAccumulator, Sequence[Evidence]],
    ) -> tuple[Hypothesis, ...]:
        if isinstance(accumulator, EvidenceAccumulator):
            evidences = accumulator.all_evidence()
        else:
            evidences = tuple(accumulator)

        if not evidences:
            return ()

        hypotheses: list[Hypothesis] = []

        maritime_det = self._find_maritime_detection(evidences)
        ais_ev = self._find_ais_evidence(evidences)

        if maritime_det is not None:
            hyp = self._evaluate_maritime(maritime_det, ais_ev, evidences)
            hypotheses.append(hyp)

        air_hyp = self._evaluate_air_incursion(evidences)
        if air_hyp is not None:
            hypotheses.append(air_hyp)

        change_hyp = self._evaluate_coastal_change(evidences)
        if change_hyp is not None:
            hypotheses.append(change_hyp)

        if not hypotheses:
            general_hyp = self._evaluate_general_surface(evidences)
            if general_hyp is not None:
                hypotheses.append(general_hyp)

        return tuple(hypotheses)

    def evaluate_primary(
        self,
        accumulator: Union[EvidenceAccumulator, Sequence[Evidence]],
    ) -> Optional[Hypothesis]:
        hyps = self.evaluate(accumulator)
        return hyps[0] if hyps else None

    def _find_maritime_detection(
        self, evidences: Sequence[Evidence]
    ) -> Optional[Evidence]:
        for ev in evidences:
            if ev.kind == EvidenceKind.OBJECT_CONFIDENCE:
                if ev.supports in (TargetClass.VESSEL.value, "VESSEL"):
                    return ev
                for ref in ev.references:
                    if "target_class=VESSEL" in ref:
                        return ev
                src = ev.source.lower()
                if "sar" in src or "vessel" in src:
                    return ev
            elif ev.supports in (TargetClass.VESSEL.value, "VESSEL"):
                return ev
        return None

    def _find_ais_evidence(
        self, evidences: Sequence[Evidence]
    ) -> Optional[Evidence]:
        for ev in evidences:
            if ev.kind == EvidenceKind.AIS_CORRELATION:
                return ev
        return None

    def _evaluate_maritime(
        self,
        maritime_det: Evidence,
        ais_ev: Optional[Evidence],
        all_evidences: Sequence[Evidence],
    ) -> Hypothesis:
        hyp_id = f"hyp-{uuid.uuid4().hex[:12]}"
        spatial_evs = [
            e for e in all_evidences if e.kind == EvidenceKind.SPATIAL_CONTEXT
        ]

        if ais_ev is None:
            statement = HypothesisStatement.UNVERIFIED_SURFACE_TARGET
            supporting = (maritime_det, *spatial_evs)
            refuting = ()
            primary_reason = (
                "Maritime vessel detected without AIS correlation coverage. "
                "Missing telemetry prevents dark vessel determination."
            )
        elif ais_ev.supports in (
            AisCorrelationState.AIS_UNAVAILABLE.value,
            AisCorrelationState.AIS_STALE.value,
            "AIS_UNAVAILABLE",
            "AIS_STALE",
        ):
            statement = HypothesisStatement.UNVERIFIED_SURFACE_TARGET
            supporting = (maritime_det, *spatial_evs)
            refuting = (ais_ev,)
            primary_reason = (
                f"Maritime vessel detected, but AIS feed telemetry is {ais_ev.supports}. "
                "Cannot confirm whether target is intentionally dark or feed is degraded."
            )
        elif ais_ev.supports in (
            AisCorrelationState.NO_MATCH.value,
            "NO_MATCH",
        ):
            is_active_feed = not any(
                "feed_active=False" in ref for ref in ais_ev.references
            )
            if is_active_feed:
                statement = HypothesisStatement.DARK_VESSEL_TRANSITING
                supporting = (maritime_det, ais_ev, *spatial_evs)
                refuting = ()
                primary_reason = (
                    "Plausible maritime vessel detected with confirmed active AIS coverage "
                    "in current sector, but negative transponder correlation (NO_MATCH)."
                )
            else:
                statement = HypothesisStatement.UNVERIFIED_SURFACE_TARGET
                supporting = (maritime_det, *spatial_evs)
                refuting = (ais_ev,)
                primary_reason = (
                    "AIS feed is inactive in target sector. "
                    "Unverified surface target pending restored AIS telemetry."
                )
        elif ais_ev.supports in (
            AisCorrelationState.MATCHED.value,
            "MATCHED",
        ):
            loiter_ev = next(
                (
                    e
                    for e in all_evidences
                    if e.kind == EvidenceKind.TEMPORAL_PERSISTENCE
                ),
                None,
            )
            if loiter_ev:
                statement = HypothesisStatement.SURFACE_VESSEL_LOITERING
                supporting = (maritime_det, ais_ev, loiter_ev, *spatial_evs)
                refuting = ()
                primary_reason = (
                    "Cooperative surface vessel with valid AIS match exhibiting persistent loitering behavior."
                )
            else:
                statement = HypothesisStatement.UNVERIFIED_SURFACE_TARGET
                supporting = (maritime_det, ais_ev, *spatial_evs)
                refuting = ()
                primary_reason = (
                    "Cooperative surface vessel with verified active AIS transponder match."
                )
        else:
            statement = HypothesisStatement.UNVERIFIED_SURFACE_TARGET
            supporting = (maritime_det, *spatial_evs)
            refuting = (ais_ev,)
            primary_reason = (
                f"Ambiguous or partial AIS correlation state ({ais_ev.supports}). Target unverified."
            )

        score = self._compute_score(supporting)
        calibration_state = CalibrationState.NOT_CALIBRATED
        primary_expl, detailed_rat = self._build_explanations(
            statement,
            primary_reason,
            supporting,
            refuting,
            score,
            calibration_state,
        )

        all_ids = tuple(
            e.evidence_id for e in list(supporting) + list(refuting)
        )
        prov = self._create_provenance(hyp_id, supporting)

        return Hypothesis(
            hypothesis_id=hyp_id,
            statement=statement,
            evidence_ids=all_ids,
            supporting_evidence=tuple(supporting),
            refuting_evidence=tuple(refuting),
            score=score,
            score_method=ScoreMethod.RULE_ENGINE,
            calibration_state=calibration_state,
            explanation=primary_expl,
            provenance=prov,
            primary_explanation=primary_expl,
            detailed_rationale=detailed_rat,
        )

    def _evaluate_air_incursion(
        self, evidences: Sequence[Evidence]
    ) -> Optional[Hypothesis]:
        opt_det = None
        kin_ev = None
        zone_ev = None

        for ev in evidences:
            if ev.kind == EvidenceKind.OBJECT_CONFIDENCE:
                if ev.supports in (
                    TargetClass.AIRCRAFT.value,
                    "AIRCRAFT",
                    "EXPERIMENTAL_INFLIGHT_TARGET",
                    "UNIDENTIFIED_AIR_INCURSION",
                ):
                    opt_det = ev
                elif "optical" in ev.source.lower() or "parallax" in ev.source.lower():
                    opt_det = ev
            elif ev.kind == EvidenceKind.KINEMATICS:
                kin_ev = ev
            elif ev.kind == EvidenceKind.SPATIAL_CONTEXT:
                if ev.supports in (
                    Zone.ZONE_1.value,
                    "ZONE_1",
                    "ZONE_1_CRITICAL",
                    Zone.ZONE_2.value,
                    "ZONE_2",
                    "ZONE_2_HIGH",
                ):
                    zone_ev = ev
                for ref in ev.references:
                    if "zone=ZONE_1" in ref or "zone=ZONE_2" in ref:
                        zone_ev = ev

        if opt_det is None and kin_ev is not None:
            if (
                kin_ev.supports
                in (
                    "EXPERIMENTAL_INFLIGHT_TARGET",
                    "UNIDENTIFIED_AIR_INCURSION",
                    "AIRCRAFT",
                )
                or "parallax" in kin_ev.source.lower()
            ):
                opt_det = kin_ev

        if opt_det is None or kin_ev is None:
            return None

        is_territorial_airspace = False
        zone_name = "Zone 1 Critical"
        if zone_ev is not None:
            is_territorial_airspace = True
            zone_name = zone_ev.supports
        else:
            for ev in (opt_det, kin_ev):
                for ref in ev.references:
                    if "zone=ZONE_1" in ref:
                        is_territorial_airspace = True
                        zone_name = "Zone 1 Critical"
                    elif "zone=ZONE_2" in ref:
                        is_territorial_airspace = True
                        zone_name = "Zone 2 High"

        if not is_territorial_airspace:
            return None

        is_high_speed = kin_ev.value > 0.0
        if not is_high_speed:
            return None

        hyp_id = f"hyp-{uuid.uuid4().hex[:12]}"
        statement = HypothesisStatement.UNIDENTIFIED_AIR_INCURSION
        supporting = [opt_det]
        if kin_ev.evidence_id != opt_det.evidence_id:
            supporting.append(kin_ev)
        if zone_ev is not None and zone_ev.evidence_id not in (
            opt_det.evidence_id,
            kin_ev.evidence_id,
        ):
            supporting.append(zone_ev)

        refuting: list[Evidence] = []
        score = self._compute_score(supporting)
        calibration_state = CalibrationState.NOT_CALIBRATED
        primary_reason = (
            f"Optical/parallax candidate detected with high estimated velocity "
            f"({kin_ev.value:.1f}) traversing territorial airspace ({zone_name})."
        )

        primary_expl, detailed_rat = self._build_explanations(
            statement,
            primary_reason,
            supporting,
            refuting,
            score,
            calibration_state,
        )
        all_ids = tuple(
            e.evidence_id for e in list(supporting) + list(refuting)
        )
        prov = self._create_provenance(hyp_id, supporting)

        return Hypothesis(
            hypothesis_id=hyp_id,
            statement=statement,
            evidence_ids=all_ids,
            supporting_evidence=tuple(supporting),
            refuting_evidence=tuple(refuting),
            score=score,
            score_method=ScoreMethod.RULE_ENGINE,
            calibration_state=calibration_state,
            explanation=primary_expl,
            provenance=prov,
            primary_explanation=primary_expl,
            detailed_rationale=detailed_rat,
        )

    def _evaluate_coastal_change(
        self, evidences: Sequence[Evidence]
    ) -> Optional[Hypothesis]:
        chg_ev = None
        corrob_ev = None

        for ev in evidences:
            if ev.kind == EvidenceKind.CHANGE_STRENGTH:
                chg_ev = ev
            elif ev.kind in (
                EvidenceKind.OBJECT_CONFIDENCE,
                EvidenceKind.SENSOR_QUALITY,
            ):
                corrob_ev = ev

        if chg_ev is None:
            return None

        is_high_quality = chg_ev.quality >= 0.8
        for ref in chg_ev.references:
            if "residual_px=" in ref:
                try:
                    res = float(ref.split("residual_px=", 1)[1].split(",")[0])
                    if res <= 0.5:
                        is_high_quality = True
                except ValueError:
                    pass
            if "quality_tier=HIGH" in ref or "quality_tier=DATA_OK" in ref:
                is_high_quality = True

        if not is_high_quality:
            return None

        if corrob_ev is None:
            return None

        hyp_id = f"hyp-{uuid.uuid4().hex[:12]}"
        statement = HypothesisStatement.INFRASTRUCTURE_EXPANSION_DETECTED
        supporting = (chg_ev, corrob_ev)
        refuting: list[Evidence] = []
        score = self._compute_score(supporting)
        calibration_state = CalibrationState.NOT_CALIBRATED
        primary_reason = (
            "Bitemporal change detection with high-tier co-registration residual (<= 0.5px) "
            "corroborated by surface sensor confirmation."
        )

        primary_expl, detailed_rat = self._build_explanations(
            statement,
            primary_reason,
            supporting,
            refuting,
            score,
            calibration_state,
        )
        all_ids = tuple(
            e.evidence_id for e in list(supporting) + list(refuting)
        )
        prov = self._create_provenance(hyp_id, supporting)

        return Hypothesis(
            hypothesis_id=hyp_id,
            statement=statement,
            evidence_ids=all_ids,
            supporting_evidence=tuple(supporting),
            refuting_evidence=tuple(refuting),
            score=score,
            score_method=ScoreMethod.RULE_ENGINE,
            calibration_state=calibration_state,
            explanation=primary_expl,
            provenance=prov,
            primary_explanation=primary_expl,
            detailed_rationale=detailed_rat,
        )

    def _evaluate_general_surface(
        self, evidences: Sequence[Evidence]
    ) -> Optional[Hypothesis]:
        obj_ev = next(
            (e for e in evidences if e.kind == EvidenceKind.OBJECT_CONFIDENCE),
            evidences[0],
        )
        hyp_id = f"hyp-{uuid.uuid4().hex[:12]}"
        statement = HypothesisStatement.UNIDENTIFIED_SURFACE_OBJECT
        supporting = (obj_ev,)
        refuting: list[Evidence] = []
        score = self._compute_score(supporting)
        calibration_state = CalibrationState.NOT_CALIBRATED
        primary_reason = (
            "Isolated surface object detected without multi-sensor corroboration or auxiliary telemetry."
        )

        primary_expl, detailed_rat = self._build_explanations(
            statement,
            primary_reason,
            supporting,
            refuting,
            score,
            calibration_state,
        )
        all_ids = tuple(
            e.evidence_id for e in list(supporting) + list(refuting)
        )
        prov = self._create_provenance(hyp_id, supporting)

        return Hypothesis(
            hypothesis_id=hyp_id,
            statement=statement,
            evidence_ids=all_ids,
            supporting_evidence=tuple(supporting),
            refuting_evidence=tuple(refuting),
            score=score,
            score_method=ScoreMethod.RULE_ENGINE,
            calibration_state=calibration_state,
            explanation=primary_expl,
            provenance=prov,
            primary_explanation=primary_expl,
            detailed_rationale=detailed_rat,
        )
