from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from environmental_os.schemas import EnvironmentalEvent


@dataclass
class ZoneTwinState:
    gps_zone: str
    waste_level: float = 0.0
    hazard_level: float = 0.0
    drain_risk: float = 0.0
    crowd_risk: float = 0.0
    last_updated: str = ""


@dataclass
class DigitalTwinState:
    """Environmental digital twin snapshot derived from fused drone intelligence."""

    zones: dict[str, ZoneTwinState] = field(default_factory=dict)

    def update_from_events(self, events: list[EnvironmentalEvent]) -> None:
        for event in events:
            zone = event.gps.rounded_zone()
            twin = self.zones.setdefault(zone, ZoneTwinState(gps_zone=zone))
            risk_map = {"low": 0.2, "medium": 0.45, "high": 0.72, "critical": 0.95}
            risk = risk_map.get(event.environmental_risk_level.value, 0.3)
            if "waste" in event.event_type.value or "garbage" in event.event_type.value:
                twin.waste_level = min(1.0, twin.waste_level + risk * 0.25)
            if event.event_type.value in {"smoke_or_fire", "chemical_spill", "toxic_leakage"}:
                twin.hazard_level = min(1.0, max(twin.hazard_level, risk))
            if "drain" in event.event_type.value or "water" in event.event_type.value:
                twin.drain_risk = min(1.0, twin.drain_risk + risk * 0.2)
            if "crowd" in event.event_type.value or "panic" in event.event_type.value:
                twin.crowd_risk = min(1.0, twin.crowd_risk + risk * 0.2)
            twin.last_updated = datetime.now(timezone.utc).isoformat()

    def snapshot(self) -> dict:
        return {
            zone: {
                "waste_level": state.waste_level,
                "hazard_level": state.hazard_level,
                "drain_risk": state.drain_risk,
                "crowd_risk": state.crowd_risk,
                "last_updated": state.last_updated,
            }
            for zone, state in self.zones.items()
        }
