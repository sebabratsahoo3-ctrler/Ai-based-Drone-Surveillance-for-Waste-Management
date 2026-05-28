from __future__ import annotations

from dataclasses import dataclass

from environmental_os.schemas import EnvironmentalEvent, GPSPoint


@dataclass(frozen=True)
class GeoFence:
    fence_id: str
    name: str
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    priority: int = 1
    sensitive: bool = False

    def contains(self, gps: GPSPoint) -> bool:
        return (
            self.min_lat <= gps.latitude <= self.max_lat
            and self.min_lon <= gps.longitude <= self.max_lon
        )


class GeoFenceMonitor:
    """Geo-fenced smart monitoring with sensitivity-aware escalation hints."""

    def __init__(self, fences: list[GeoFence] | None = None):
        self.fences = fences or []

    def active_fences(self, gps: GPSPoint) -> list[GeoFence]:
        return [fence for fence in self.fences if fence.contains(gps)]

    def annotate_event(self, event: EnvironmentalEvent) -> EnvironmentalEvent:
        fences = self.active_fences(event.gps)
        if not fences:
            return event
        event.metadata["geo_fences"] = [fence.fence_id for fence in fences]
        if any(fence.sensitive for fence in fences):
            event.metadata["sensitive_zone"] = True
            if event.confidence >= 0.5:
                event.confidence = min(0.99, event.confidence + 0.05)
                event.rationale = (
                    f"{event.rationale} Event occurred inside a sensitive geo-fenced zone."
                ).strip()
        return event
