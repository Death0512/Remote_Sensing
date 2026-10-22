from typing import Union
from src.domain.enums import AlertPriority, HypothesisStatement, Zone
from src.domain.hypothesis import Hypothesis

CRITICAL_CONJUNCTIONS = {
    HypothesisStatement.DARK_VESSEL_TRANSITING,
    HypothesisStatement.UNIDENTIFIED_AIR_INCURSION,
}


def _extract_statement(hypothesis: Union[Hypothesis, HypothesisStatement, str]) -> HypothesisStatement:
    if isinstance(hypothesis, Hypothesis):
        return hypothesis.statement
    if isinstance(hypothesis, HypothesisStatement):
        return hypothesis
    return HypothesisStatement(hypothesis)


def _extract_zone(zone: Union[Zone, str]) -> Zone:
    if isinstance(zone, Zone):
        return zone
    return Zone(zone)


def determine_alert_priority(
    zone: Union[Zone, str],
    hypothesis: Union[Hypothesis, HypothesisStatement, str],
) -> AlertPriority:
    st = _extract_statement(hypothesis)
    zn = _extract_zone(zone)
    is_conjunction = st in CRITICAL_CONJUNCTIONS

    if zn in (Zone.ZONE_1, Zone.ZONE_1_CRITICAL):
        return AlertPriority.CRITICAL if is_conjunction else AlertPriority.HIGH
    elif zn in (Zone.ZONE_2, Zone.ZONE_2_HIGH):
        return AlertPriority.HIGH if is_conjunction else AlertPriority.MEDIUM
    elif zn in (Zone.ZONE_3, Zone.ZONE_3_MEDIUM):
        return AlertPriority.MEDIUM if is_conjunction else AlertPriority.LOW
    else:
        return AlertPriority.INFORMATIONAL


def validate_priority_gating(
    priority: Union[AlertPriority, str],
    zone: Union[Zone, str],
    hypothesis: Union[Hypothesis, HypothesisStatement, str],
) -> None:
    p = AlertPriority(priority) if isinstance(priority, str) else priority
    zn = _extract_zone(zone)
    st = _extract_statement(hypothesis)

    if p == AlertPriority.CRITICAL:
        if zn not in (Zone.ZONE_1, Zone.ZONE_1_CRITICAL) or st not in CRITICAL_CONJUNCTIONS:
            raise ValueError(
                f"AlertPriority.CRITICAL requires Zone 1 and verified conjunction ({st.value} in {zn.value} invalid)"
            )
