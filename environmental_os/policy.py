from __future__ import annotations

from .schemas import EnvironmentalRisk, EscalationStatus, EventType, Severity


MAJOR_EVENTS = {
    EventType.TOXIC_LEAKAGE,
    EventType.SMOKE_OR_FIRE,
    EventType.HAZARDOUS_WASTE_EXPOSURE,
    EventType.WASTE_BURNING,
    EventType.CHEMICAL_SPILL,
    EventType.SEWAGE_OVERFLOW,
    EventType.LARGE_ILLEGAL_DUMPING,
    EventType.FALLEN_ELECTRIC_POLE,
    EventType.PUBLIC_PANIC,
}

MODERATE_EVENTS = {
    EventType.OVERFLOWING_BIN,
    EventType.REPEATED_SANITATION_VIOLATION,
    EventType.PUBLIC_SPITTING,
    EventType.GARBAGE_DUMPING,
    EventType.DRAIN_BLOCKAGE,
    EventType.NIGHT_DUMPING,
    EventType.CONSTRUCTION_DEBRIS_DUMPING,
    EventType.DRAIN_OR_RIVER_DUMPING,
    EventType.VEHICLE_DUMPING,
    EventType.WATERLOGGING,
    EventType.CROWD_NEAR_HAZARD,
    EventType.ENVIRONMENTAL_CONTAMINATION,
    EventType.ROAD_BLOCKAGE,
    EventType.WASTE_CAUSED_ACCIDENT,
    EventType.UNUSUAL_CROWD_BEHAVIOR,
    EventType.SENSITIVE_ZONE_ACCUMULATION,
}

MINOR_EVENTS = {
    EventType.ROADSIDE_LITTERING,
    EventType.OUTSIDE_BIN_WASTE,
    EventType.TEMPORARY_WASTE_ACCUMULATION,
    EventType.ANIMALS_SCAVENGING,
    EventType.PUBLIC_URINATION,
}


def classify_severity(
    event_type: EventType,
    confidence: float,
    historical_relevance_score: float,
) -> Severity:
    if event_type in MAJOR_EVENTS:
        return Severity.MAJOR
    if historical_relevance_score >= 0.72 and confidence >= 0.62:
        return Severity.MODERATE
    if event_type in MODERATE_EVENTS:
        return Severity.MODERATE
    return Severity.MINOR


def estimate_risk(
    severity: Severity,
    confidence: float,
    historical_relevance_score: float,
) -> EnvironmentalRisk:
    score = (confidence * 0.65) + (historical_relevance_score * 0.35)
    if severity == Severity.MAJOR and score >= 0.58:
        return EnvironmentalRisk.CRITICAL
    if severity == Severity.MAJOR:
        return EnvironmentalRisk.HIGH
    if severity == Severity.MODERATE and score >= 0.68:
        return EnvironmentalRisk.HIGH
    if severity == Severity.MODERATE:
        return EnvironmentalRisk.MEDIUM
    if score >= 0.74:
        return EnvironmentalRisk.MEDIUM
    return EnvironmentalRisk.LOW


def decide_escalation(
    severity: Severity,
    risk: EnvironmentalRisk,
    confidence: float,
    historical_relevance_score: float,
) -> EscalationStatus:
    if risk == EnvironmentalRisk.CRITICAL and confidence >= 0.55:
        return EscalationStatus.AUTO_DISPATCH
    if severity == Severity.MAJOR:
        return EscalationStatus.COMMAND_REVIEW
    if historical_relevance_score >= 0.7 and severity == Severity.MODERATE:
        return EscalationStatus.COMMAND_REVIEW
    if severity == Severity.MODERATE:
        return EscalationStatus.WATCHLIST
    return EscalationStatus.LOG_ONLY
