from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from environmental_os.schemas import EnvironmentalEvent


@dataclass
class ZoneForecast:
    gps_zone: str
    latitude: float
    longitude: float
    recent_event_count: int
    trend: str
    forecast_score: float
    recommendation: str


class WasteAccumulationForecast:
    """Simple hotspot forecasting from recent event density and severity."""

    def forecast(self, events: list[EnvironmentalEvent], window_hours: int = 72) -> list[ZoneForecast]:
        zones: dict[str, list[EnvironmentalEvent]] = defaultdict(list)
        for event in events:
            zones[event.gps.rounded_zone()].append(event)

        forecasts: list[ZoneForecast] = []
        for zone, zone_events in zones.items():
            major = sum(1 for e in zone_events if e.severity.value == "Major Trigger")
            moderate = sum(1 for e in zone_events if e.severity.value == "Moderate Trigger")
            count = len(zone_events)
            score = min(1.0, (major * 0.35) + (moderate * 0.2) + (count * 0.08))
            trend = "rising" if count >= 3 else "stable" if count >= 2 else "low"
            recommendation = self._recommend(score, trend)
            forecasts.append(
                ZoneForecast(
                    gps_zone=zone,
                    latitude=zone_events[-1].gps.latitude,
                    longitude=zone_events[-1].gps.longitude,
                    recent_event_count=count,
                    trend=trend,
                    forecast_score=round(score, 3),
                    recommendation=recommendation,
                )
            )
        return sorted(forecasts, key=lambda item: item.forecast_score, reverse=True)

    @staticmethod
    def _recommend(score: float, trend: str) -> str:
        if score >= 0.75:
            return "Increase collection frequency and schedule focused nighttime patrol."
        if score >= 0.45 and trend == "rising":
            return "Monitor zone closely; waste accumulation trend is increasing."
        if score >= 0.45:
            return "Maintain routine patrol; moderate accumulation risk."
        return "Continue standard patrol cadence."
