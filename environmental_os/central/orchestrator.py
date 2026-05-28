from __future__ import annotations

import uuid
from datetime import datetime, timezone

from environmental_os.central.swarm import SwarmCoordinator
from environmental_os.db.repository import EnvironmentalDatabase
from environmental_os.fusion.multimodal import MultiModalFusionEngine, SensorBundle
from environmental_os.geo.fence import GeoFenceMonitor
from environmental_os.intelligence.digital_twin import DigitalTwinState
from environmental_os.intelligence.forecast import WasteAccumulationForecast
from environmental_os.policy import classify_severity, decide_escalation, estimate_risk
from environmental_os.rag.vector_store import HistoricalRecord, LocalVectorStore
from environmental_os.schemas import (
    EnvironmentalEvent,
    FrameContext,
    SceneObservation,
)
from environmental_os.vlm.models import VLMReasoner
from environmental_os.vlm.temporal import TemporalEventReasoner


class EnvironmentalOrchestrator:
    def __init__(
        self,
        vlm: VLMReasoner,
        vector_store: LocalVectorStore | None = None,
        temporal_reasoner: TemporalEventReasoner | None = None,
        fusion_engine: MultiModalFusionEngine | None = None,
        geofence_monitor: GeoFenceMonitor | None = None,
        swarm: SwarmCoordinator | None = None,
        database: EnvironmentalDatabase | None = None,
    ):
        self.vlm = vlm
        self.vector_store = vector_store or LocalVectorStore()
        self.temporal_reasoner = temporal_reasoner or TemporalEventReasoner()
        self.fusion_engine = fusion_engine or MultiModalFusionEngine()
        self.geofence_monitor = geofence_monitor or GeoFenceMonitor()
        self.swarm = swarm or SwarmCoordinator()
        self.database = database
        self.forecast_engine = WasteAccumulationForecast()
        self.digital_twin = DigitalTwinState()
        self.events: list[EnvironmentalEvent] = []

    def process_frame(self, frame: FrameContext, sensors: SensorBundle | None = None) -> list[EnvironmentalEvent]:
        zone = frame.telemetry.gps.rounded_zone()
        query = self._query_from_frame(frame)
        historical = self.vector_store.search(query=query, gps_zone=zone, limit=4)
        historical_score = self._history_score(historical, zone)
        rag_context = [
            f"{'same_zone' if item.record.gps_zone == zone else 'related_zone'} "
            f"{item.record.timestamp} {item.record.event_type} {item.record.text}"
            for item in historical
            if item.score >= 0.15
        ]
        temporal_context = self.temporal_reasoner.context_for(frame)
        observation = self.vlm.analyze(frame, rag_context, temporal_context)
        sensor_bundle = sensors or SensorBundle(frame=frame)
        # Ensure the sensor bundle contains the same historical score the
        # orchestrator computed for this GPS zone.
        sensor_bundle.frame = frame
        sensor_bundle.historical_hotspot_score = historical_score
        observation.candidates = self.fusion_engine.fuse(observation.candidates, sensor_bundle)
        observation.candidates = self.temporal_reasoner.stabilize(observation)
        events = self._promote_candidates(observation, historical_score=historical_score)
        events = [self.geofence_monitor.annotate_event(event) for event in events]
        self._correlate(events)
        events = self.swarm.ingest(events)
        self._persist_to_memory(events)
        self.events.extend(events)
        self.digital_twin.update_from_events(events)
        if self.database:
            self.database.save_events(events)
        return events

    def ingest_observation(self, observation: SceneObservation) -> list[EnvironmentalEvent]:
        observation.candidates = self.temporal_reasoner.stabilize(observation)
        events = self._promote_candidates(observation, historical_score=0.0)
        events = [self.geofence_monitor.annotate_event(event) for event in events]
        self._correlate(events)
        events = self.swarm.ingest(events)
        self._persist_to_memory(events)
        self.events.extend(events)
        self.digital_twin.update_from_events(events)
        if self.database:
            self.database.save_events(events)
        return events

    def heatmap(self) -> list[dict]:
        zones: dict[str, dict] = {}
        for event in self.events:
            zone = event.gps.rounded_zone()
            item = zones.setdefault(
                zone,
                {
                    "zone": zone,
                    "latitude": event.gps.latitude,
                    "longitude": event.gps.longitude,
                    "count": 0,
                    "max_risk": event.environmental_risk_level.value,
                    "major": 0,
                    "moderate": 0,
                    "minor": 0,
                },
            )
            item["count"] += 1
            key = event.severity.name.lower()
            item[key] += 1
            item["max_risk"] = self._higher_risk(item["max_risk"], event.environmental_risk_level.value)
        return sorted(zones.values(), key=lambda item: item["count"], reverse=True)

    def _promote_candidates(
        self,
        observation: SceneObservation,
        historical_score: float,
    ) -> list[EnvironmentalEvent]:
        events = []
        frame = observation.frame
        for candidate in observation.candidates:
            candidate_history = max(historical_score, 0.78 if candidate.metadata.get("repeated_zone_signal") else 0.0)
            severity = classify_severity(candidate.event_type, candidate.confidence, candidate_history)
            risk = estimate_risk(severity, candidate.confidence, candidate_history)
            escalation = decide_escalation(severity, risk, candidate.confidence, candidate_history)
            events.append(
                EnvironmentalEvent(
                    event_id=f"evt-{uuid.uuid4().hex[:12]}",
                    timestamp=frame.telemetry.timestamp,
                    gps=frame.telemetry.gps,
                    event_type=candidate.event_type,
                    confidence=round(candidate.confidence, 3),
                    severity=severity,
                    snapshot_ref=candidate.snapshot_ref or frame.frame_ref,
                    drone_altitude_m=frame.telemetry.altitude_m,
                    drone_id=frame.telemetry.drone_id,
                    scene_description=candidate.scene_description,
                    historical_relevance_score=round(candidate_history, 3),
                    environmental_risk_level=risk,
                    escalation_status=escalation,
                    involved_agents=candidate.involved_agents,
                    rationale=candidate.rationale,
                    temporal_evidence=candidate.temporal_evidence,
                    metadata={
                        "vlm_model": observation.vlm_model,
                        "processing_ms": observation.processing_ms,
                        "camera_id": frame.camera_id,
                        "weather": frame.weather,
                        "lighting": frame.lighting,
                        **candidate.metadata,
                    },
                )
            )
        return events

    def _correlate(self, new_events: list[EnvironmentalEvent]) -> None:
        for event in new_events:
            for previous in reversed(self.events[-50:]):
                same_zone = previous.gps.rounded_zone() == event.gps.rounded_zone()
                same_type = previous.event_type == event.event_type
                if same_zone and (same_type or previous.severity == event.severity):
                    event.correlated_event_ids.append(previous.event_id)
                    if len(event.correlated_event_ids) >= 5:
                        break

    def _persist_to_memory(self, events: list[EnvironmentalEvent]) -> None:
        for event in events:
            self.vector_store.add(
                HistoricalRecord(
                    record_id=event.event_id,
                    timestamp=event.timestamp,
                    gps_zone=event.gps.rounded_zone(),
                    text=(
                        f"{event.scene_description} Risk {event.environmental_risk_level.value}. "
                        f"Escalation {event.escalation_status.value}. {event.rationale}"
                    ),
                    event_type=event.event_type.value,
                    severity=event.severity.value,
                    metadata={"drone_id": event.drone_id},
                )
            )

    @staticmethod
    def _query_from_frame(frame: FrameContext) -> str:
        return (
            f"{frame.telemetry.gps.rounded_zone()} {frame.lighting} {frame.weather} "
            f"{frame.metadata.get('scene_hint', '')} {' '.join(frame.roi_hints)}"
        )

    @staticmethod
    def _history_score(items, zone: str) -> float:
        if not items:
            return 0.0
        score = max(item.score for item in items)
        same_zone_seen = any(item.record.gps_zone == zone for item in items)
        if same_zone_seen:
            score = max(score, 0.72)
        return min(score, 1.0)

    @staticmethod
    def _higher_risk(left: str, right: str) -> str:
        order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        return left if order[left] >= order[right] else right

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def forecast(self) -> list[dict]:
        return [item.__dict__ for item in self.forecast_engine.forecast(self.events)]

    def digital_twin_snapshot(self) -> dict:
        return self.digital_twin.snapshot()
