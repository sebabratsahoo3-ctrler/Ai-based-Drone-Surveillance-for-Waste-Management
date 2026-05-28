from __future__ import annotations

from dataclasses import dataclass, field

from environmental_os.schemas import EnvironmentalEvent, GPSPoint


@dataclass
class DroneFleetState:
    drone_id: str
    active: bool = True
    battery_pct: float = 100.0
    last_gps: GPSPoint | None = None
    events_today: int = 0


@dataclass
class SwarmCoordinator:
    """Multi-drone coordination, deduplication, and patrol load balancing."""

    fleet: dict[str, DroneFleetState] = field(default_factory=dict)
    dedup_radius_m: float = 35.0

    def register(self, drone_id: str) -> DroneFleetState:
        if drone_id not in self.fleet:
            self.fleet[drone_id] = DroneFleetState(drone_id=drone_id)
        return self.fleet[drone_id]

    def ingest(self, events: list[EnvironmentalEvent]) -> list[EnvironmentalEvent]:
        merged: list[EnvironmentalEvent] = []
        for event in events:
            state = self.register(event.drone_id)
            state.last_gps = event.gps
            state.events_today += 1
            if not self._is_duplicate(event, merged):
                merged.append(event)
        return merged

    def assign_patrol_priority(self) -> list[str]:
        """Return drone IDs sorted by lowest recent activity for rebalancing."""
        return sorted(
            self.fleet.keys(),
            key=lambda drone_id: (self.fleet[drone_id].events_today, -self.fleet[drone_id].battery_pct),
        )

    def _is_duplicate(self, event: EnvironmentalEvent, existing: list[EnvironmentalEvent]) -> bool:
        for other in existing:
            if other.event_type != event.event_type:
                continue
            if other.gps.rounded_zone() == event.gps.rounded_zone():
                if abs(other.confidence - event.confidence) < 0.12:
                    event.correlated_event_ids.append(other.event_id)
                    return True
        return False
