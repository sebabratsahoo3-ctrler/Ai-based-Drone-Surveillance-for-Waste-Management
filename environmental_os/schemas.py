from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Severity(str, Enum):
    MAJOR = "Major Trigger"
    MODERATE = "Moderate Trigger"
    MINOR = "Minor Trigger"


class EnvironmentalRisk(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EscalationStatus(str, Enum):
    AUTO_DISPATCH = "auto_dispatch"
    COMMAND_REVIEW = "command_review"
    WATCHLIST = "watchlist"
    LOG_ONLY = "log_only"


class EventType(str, Enum):
    GARBAGE_DUMPING = "garbage_dumping"
    ROADSIDE_LITTERING = "roadside_littering"
    PUBLIC_SPITTING = "public_spitting"
    OUTSIDE_BIN_WASTE = "outside_bin_waste"
    DRAIN_OR_RIVER_DUMPING = "drain_or_river_dumping"
    CONSTRUCTION_DEBRIS_DUMPING = "construction_debris_dumping"
    OVERFLOWING_BIN = "overflowing_bin"
    PUBLIC_URINATION = "public_urination"
    WASTE_BURNING = "waste_burning"
    SMOKE_OR_FIRE = "smoke_or_fire"
    TOXIC_LEAKAGE = "toxic_leakage"
    SEWAGE_OVERFLOW = "sewage_overflow"
    ENVIRONMENTAL_CONTAMINATION = "environmental_contamination"
    DRAIN_BLOCKAGE = "drain_blockage"
    WATERLOGGING = "waterlogging"
    VEHICLE_DUMPING = "vehicle_dumping"
    LARGE_ILLEGAL_DUMPING = "large_illegal_dumping"
    NIGHT_DUMPING = "night_dumping"
    REPEATED_SANITATION_VIOLATION = "repeated_sanitation_violation"
    CROWD_NEAR_HAZARD = "crowd_near_hazard"
    ANIMALS_SCAVENGING = "animals_scavenging"
    HAZARDOUS_WASTE_EXPOSURE = "hazardous_waste_exposure"
    SENSITIVE_ZONE_ACCUMULATION = "sensitive_zone_accumulation"
    CHEMICAL_SPILL = "chemical_spill"
    ROAD_BLOCKAGE = "road_blockage"
    WASTE_CAUSED_ACCIDENT = "waste_caused_accident"
    FALLEN_ELECTRIC_POLE = "fallen_electric_pole"
    PUBLIC_PANIC = "public_panic"
    UNUSUAL_CROWD_BEHAVIOR = "unusual_crowd_behavior"
    TEMPORARY_WASTE_ACCUMULATION = "temporary_waste_accumulation"


@dataclass(frozen=True)
class GPSPoint:
    latitude: float
    longitude: float

    def rounded_zone(self, precision: int = 4) -> str:
        return f"{round(self.latitude, precision)}:{round(self.longitude, precision)}"


@dataclass
class DroneTelemetry:
    drone_id: str
    timestamp: str
    gps: GPSPoint
    altitude_m: float
    speed_mps: float = 0.0
    heading_deg: float = 0.0
    battery_pct: float = 100.0
    mission_id: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def now(
        cls,
        drone_id: str,
        gps: GPSPoint,
        altitude_m: float,
        mission_id: str = "default",
    ) -> "DroneTelemetry":
        return cls(
            drone_id=drone_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            gps=gps,
            altitude_m=altitude_m,
            mission_id=mission_id,
        )


@dataclass
class FrameContext:
    frame_id: str
    frame_ref: str
    telemetry: DroneTelemetry
    camera_id: str = "front_center"
    weather: str = "clear"
    lighting: str = "day"
    resolution: tuple[int, int] = (640, 360)
    roi_hints: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventCandidate:
    event_type: EventType
    confidence: float
    scene_description: str
    involved_agents: list[str] = field(default_factory=list)
    rationale: str = ""
    temporal_evidence: list[str] = field(default_factory=list)
    snapshot_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentalEvent:
    event_id: str
    timestamp: str
    gps: GPSPoint
    event_type: EventType
    confidence: float
    severity: Severity
    snapshot_ref: str
    drone_altitude_m: float
    drone_id: str
    scene_description: str
    historical_relevance_score: float
    environmental_risk_level: EnvironmentalRisk
    escalation_status: EscalationStatus
    involved_agents: list[str] = field(default_factory=list)
    rationale: str = ""
    temporal_evidence: list[str] = field(default_factory=list)
    correlated_event_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["severity"] = self.severity.value
        data["environmental_risk_level"] = self.environmental_risk_level.value
        data["escalation_status"] = self.escalation_status.value
        return data


@dataclass
class SceneObservation:
    frame: FrameContext
    summary: str
    candidates: list[EventCandidate]
    vlm_model: str
    processing_ms: float = 0.0
    raw_reasoning: dict[str, Any] = field(default_factory=dict)


@dataclass
class IncidentReport:
    report_id: str
    generated_at: str
    title: str
    summary: str
    events: list[EnvironmentalEvent]
    recommendations: list[str]
    dispatch_targets: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["events"] = [event.to_dict() for event in self.events]
        return data
