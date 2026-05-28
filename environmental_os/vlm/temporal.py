from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from environmental_os.schemas import EventCandidate, EventType, FrameContext, SceneObservation


@dataclass
class TemporalMemory:
    max_frames: int = 12
    frames_by_drone: dict[str, deque[FrameContext]] = field(default_factory=lambda: defaultdict(deque))
    recent_candidates_by_zone: dict[str, deque[EventCandidate]] = field(default_factory=lambda: defaultdict(deque))

    def remember_frame(self, frame: FrameContext) -> None:
        bucket = self.frames_by_drone[frame.telemetry.drone_id]
        bucket.append(frame)
        while len(bucket) > self.max_frames:
            bucket.popleft()

    def remember_candidates(self, frame: FrameContext, candidates: list[EventCandidate]) -> None:
        zone = frame.telemetry.gps.rounded_zone()
        bucket = self.recent_candidates_by_zone[zone]
        for candidate in candidates:
            bucket.append(candidate)
        while len(bucket) > self.max_frames * 3:
            bucket.popleft()

    def frame_context(self, drone_id: str) -> list[str]:
        frames = self.frames_by_drone.get(drone_id, [])
        return [
            f"{frame.telemetry.timestamp} {frame.frame_id} "
            f"{frame.metadata.get('scene_hint', '')}".strip()
            for frame in frames
        ]

    def zone_context(self, frame: FrameContext) -> list[str]:
        zone = frame.telemetry.gps.rounded_zone()
        candidates = self.recent_candidates_by_zone.get(zone, [])
        return [
            f"{candidate.event_type.value}: {candidate.scene_description}"
            for candidate in candidates
        ]


class TemporalEventReasoner:
    def __init__(self, memory: TemporalMemory | None = None):
        self.memory = memory or TemporalMemory()

    def context_for(self, frame: FrameContext) -> list[str]:
        return self.memory.frame_context(frame.telemetry.drone_id) + self.memory.zone_context(frame)

    def stabilize(self, observation: SceneObservation) -> list[EventCandidate]:
        frame = observation.frame
        zone_context = self.memory.zone_context(frame)
        stabilized: list[EventCandidate] = []

        for candidate in observation.candidates:
            candidate.temporal_evidence.extend(zone_context[-3:])
            repeated_same_zone = any(candidate.event_type.value in item for item in zone_context)
            repeated_sanitation = any(
                token in item
                for token in ("litter", "dump", "overflowing_bin", "drain_blockage")
                for item in zone_context
            )

            if repeated_same_zone:
                candidate.confidence = min(0.98, candidate.confidence + 0.08)
                candidate.rationale = f"{candidate.rationale} Similar incident persisted across recent frames.".strip()
            if repeated_sanitation and candidate.event_type not in {
                EventType.SMOKE_OR_FIRE,
                EventType.CHEMICAL_SPILL,
                EventType.TOXIC_LEAKAGE,
            }:
                candidate.metadata["repeated_zone_signal"] = True

            stabilized.append(candidate)

        if self._should_add_repeated_violation(observation, zone_context):
            stabilized.append(
                EventCandidate(
                    event_type=EventType.REPEATED_SANITATION_VIOLATION,
                    confidence=0.77,
                    scene_description="Repeated sanitation violations detected in this GPS zone.",
                    rationale="Multiple recent frames or events show recurring sanitation issues at the same location.",
                    temporal_evidence=zone_context[-5:],
                    snapshot_ref=frame.frame_ref,
                )
            )

        self.memory.remember_frame(frame)
        self.memory.remember_candidates(frame, stabilized)
        return stabilized

    @staticmethod
    def _should_add_repeated_violation(observation: SceneObservation, zone_context: list[str]) -> bool:
        sanitation_candidates = [
            item
            for item in observation.candidates
            if item.event_type in {
                EventType.GARBAGE_DUMPING,
                EventType.ROADSIDE_LITTERING,
                EventType.OVERFLOWING_BIN,
                EventType.DRAIN_BLOCKAGE,
                EventType.DRAIN_OR_RIVER_DUMPING,
                EventType.TEMPORARY_WASTE_ACCUMULATION,
            }
        ]
        sanitation_history = [
            item
            for item in zone_context
            if any(token in item for token in ("dump", "litter", "overflowing_bin", "drain_blockage", "waste"))
        ]
        return bool(sanitation_candidates and len(sanitation_history) >= 2)

