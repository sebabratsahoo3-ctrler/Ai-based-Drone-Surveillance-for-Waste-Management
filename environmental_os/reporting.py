from __future__ import annotations

import uuid
from datetime import datetime, timezone

from environmental_os.schemas import EnvironmentalEvent, EscalationStatus, IncidentReport, Severity


def build_incident_report(events: list[EnvironmentalEvent]) -> IncidentReport:
    major = [event for event in events if event.severity == Severity.MAJOR]
    moderate = [event for event in events if event.severity == Severity.MODERATE]
    dispatch = [event for event in events if event.escalation_status == EscalationStatus.AUTO_DISPATCH]

    if major:
        title = "Critical environmental hazards require immediate review"
    elif moderate:
        title = "Recurring sanitation and environmental risks detected"
    elif events:
        title = "Minor sanitation observations logged"
    else:
        title = "No active environmental incidents"

    summary = _summary(events, major, moderate)
    recommendations = _recommendations(events)
    dispatch_targets = sorted(
        {
            _dispatch_target(event)
            for event in dispatch or major
        }
    )

    return IncidentReport(
        report_id=f"rpt-{uuid.uuid4().hex[:10]}",
        generated_at=datetime.now(timezone.utc).isoformat(),
        title=title,
        summary=summary,
        events=events,
        recommendations=recommendations,
        dispatch_targets=dispatch_targets,
    )


def _summary(
    events: list[EnvironmentalEvent],
    major: list[EnvironmentalEvent],
    moderate: list[EnvironmentalEvent],
) -> str:
    if not events:
        return "No events have been detected in the current monitoring window."
    zones = len({event.gps.rounded_zone() for event in events})
    return (
        f"{len(events)} events detected across {zones} GPS zones. "
        f"{len(major)} major triggers and {len(moderate)} moderate triggers require attention. "
        f"Most recent event: {events[-1].scene_description}"
    )


def _recommendations(events: list[EnvironmentalEvent]) -> list[str]:
    recommendations: list[str] = []
    event_types = {event.event_type.value for event in events}
    if {"smoke_or_fire", "waste_burning"} & event_types:
        recommendations.append("Dispatch fire and municipal response teams to verify smoke or open burning.")
    if {"chemical_spill", "toxic_leakage", "hazardous_waste_exposure"} & event_types:
        recommendations.append("Create a hazard perimeter and send trained environmental safety personnel.")
    if {"drain_blockage", "waterlogging", "drain_or_river_dumping"} & event_types:
        recommendations.append("Prioritize drain clearance and water contamination inspection.")
    if {"overflowing_bin", "sensitive_zone_accumulation"} & event_types:
        recommendations.append("Increase collection frequency near sensitive public zones.")
    if {"night_dumping", "vehicle_dumping", "large_illegal_dumping"} & event_types:
        recommendations.append("Schedule nighttime patrol reinforcement and vehicle evidence review.")
    if not recommendations and events:
        recommendations.append("Log incident and continue patrol monitoring for recurrence.")
    return recommendations


def _dispatch_target(event: EnvironmentalEvent) -> str:
    if event.event_type.value in {"smoke_or_fire", "waste_burning"}:
        return "fire_response"
    if event.event_type.value in {"chemical_spill", "toxic_leakage", "hazardous_waste_exposure"}:
        return "environmental_hazard_team"
    if event.event_type.value in {"sewage_overflow", "drain_blockage", "waterlogging"}:
        return "drainage_and_sanitation_team"
    return "municipal_sanitation_team"
