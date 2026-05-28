from __future__ import annotations

from dataclasses import dataclass, field

from environmental_os.schemas import EventCandidate, EventType, FrameContext


@dataclass
class SensorBundle:
    frame: FrameContext
    audio_anomaly_score: float = 0.0
    audio_labels: list[str] = field(default_factory=list)
    thermal_hotspot_score: float = 0.0
    thermal_labels: list[str] = field(default_factory=list)
    weather_risk: float = 0.0
    crowd_density: float = 0.0
    historical_hotspot_score: float = 0.0


class MultiModalFusionEngine:
    """Fuse vision VLM candidates with audio, thermal, GPS, weather, and history."""

    def fuse(self, candidates: list[EventCandidate], sensors: SensorBundle) -> list[EventCandidate]:
        fused: list[EventCandidate] = []
        for candidate in candidates:
            adjusted = self._adjust_candidate(candidate, sensors)
            fused.append(adjusted)
        fused.extend(self._audio_thermal_candidates(sensors))
        return fused

    def _adjust_candidate(self, candidate: EventCandidate, sensors: SensorBundle) -> EventCandidate:
        boost = 0.0
        rationale_parts = [candidate.rationale]

        if sensors.historical_hotspot_score >= 0.65:
            boost += 0.06
            rationale_parts.append("GPS zone has prior sanitation hotspot history.")

        if sensors.weather_risk >= 0.5 and candidate.event_type in {
            EventType.WATERLOGGING,
            EventType.DRAIN_BLOCKAGE,
            EventType.ENVIRONMENTAL_CONTAMINATION,
        }:
            boost += 0.05
            rationale_parts.append("Weather conditions increase contamination and flooding risk.")

        if sensors.crowd_density >= 0.6 and candidate.event_type in {
            EventType.CROWD_NEAR_HAZARD,
            EventType.PUBLIC_PANIC,
            EventType.UNUSUAL_CROWD_BEHAVIOR,
        }:
            boost += 0.07

        if sensors.thermal_hotspot_score >= 0.7 and candidate.event_type in {
            EventType.SMOKE_OR_FIRE,
            EventType.WASTE_BURNING,
        }:
            boost += 0.1
            rationale_parts.append("Thermal hotspot corroborates smoke or fire interpretation.")

        if sensors.audio_anomaly_score >= 0.65:
            if any(label in sensors.audio_labels for label in ("explosion", "alarm", "shouting")):
                if candidate.event_type == EventType.PUBLIC_PANIC:
                    boost += 0.08
            if "crackling" in sensors.audio_labels or "siren" in sensors.audio_labels:
                if candidate.event_type in {EventType.SMOKE_OR_FIRE, EventType.WASTE_BURNING}:
                    boost += 0.09
                    rationale_parts.append("Audio anomaly supports combustion or emergency cues.")

        candidate.confidence = min(0.99, candidate.confidence + boost)
        candidate.rationale = " ".join(part for part in rationale_parts if part).strip()
        candidate.metadata["multimodal_fusion"] = {
            "audio_score": sensors.audio_anomaly_score,
            "thermal_score": sensors.thermal_hotspot_score,
            "weather_risk": sensors.weather_risk,
            "historical_hotspot": sensors.historical_hotspot_score,
        }
        return candidate

    def _audio_thermal_candidates(self, sensors: SensorBundle) -> list[EventCandidate]:
        extra: list[EventCandidate] = []
        if sensors.thermal_hotspot_score >= 0.82 and not sensors.frame.metadata.get("scene_hint", "").lower().count("fire"):
            extra.append(
                EventCandidate(
                    event_type=EventType.SMOKE_OR_FIRE,
                    confidence=min(0.88, 0.55 + sensors.thermal_hotspot_score * 0.35),
                    scene_description="Thermal hazard detected independent of visible smoke plume.",
                    rationale="Thermal sensor indicates elevated temperature consistent with fire risk.",
                    snapshot_ref=sensors.frame.frame_ref,
                    metadata={"source": "thermal_only"},
                )
            )
        if sensors.audio_anomaly_score >= 0.8 and "shouting" in sensors.audio_labels:
            extra.append(
                EventCandidate(
                    event_type=EventType.PUBLIC_PANIC,
                    confidence=0.72,
                    scene_description="Unusual crowd audio patterns detected near monitored zone.",
                    rationale="Audio anomaly suggests public distress or panic.",
                    involved_agents=["crowd"],
                    snapshot_ref=sensors.frame.frame_ref,
                    metadata={"source": "audio_only"},
                )
            )
        return extra
