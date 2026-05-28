from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

from environmental_os.airsim_bridge.client import AirSimDroneBridge
from environmental_os.drone.frame_buffer import FrameBuffer
from environmental_os.schemas import FrameContext, GPSPoint


@dataclass
class PatrolWaypoint:
    latitude: float
    longitude: float
    altitude_m: float = 42.0
    scene_hint: str = ""
    lighting: str = "day"
    weather: str = "clear"


@dataclass
class DronePatrolUnit:
    """Autonomous aerial patrol unit with GPS tagging and frame buffering."""

    drone_id: str
    bridge: AirSimDroneBridge | None = None
    frame_buffer: FrameBuffer = field(default_factory=FrameBuffer)
    mission_id: str = "sanitation-patrol"
    connected: bool = False
    _on_frame: list[Callable[[FrameContext], None]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.bridge is None:
            self.bridge = AirSimDroneBridge(self.drone_id)

    def connect(self) -> bool:
        self.connected = self.bridge.connect()
        return self.connected

    def on_frame(self, handler: Callable[[FrameContext], None]) -> None:
        self._on_frame.append(handler)

    def capture(
        self,
        scene_hint: str,
        gps: GPSPoint | None = None,
        altitude_m: float = 42.0,
        lighting: str = "day",
        weather: str = "clear",
        roi_hints: list[str] | None = None,
    ) -> FrameContext:
        frame = self.bridge.mock_frame(
            scene_hint=scene_hint,
            gps=gps,
            altitude_m=altitude_m,
            lighting=lighting,
        )
        frame.weather = weather
        frame.roi_hints = list(roi_hints or [])
        frame.telemetry.mission_id = self.mission_id
        self.frame_buffer.push(frame)
        for handler in self._on_frame:
            handler(frame)
        return frame

    def patrol_route(self, waypoints: list[PatrolWaypoint]) -> Iterator[FrameContext]:
        for wp in waypoints:
            yield self.capture(
                scene_hint=wp.scene_hint,
                gps=GPSPoint(wp.latitude, wp.longitude),
                altitude_m=wp.altitude_m,
                lighting=wp.lighting,
                weather=wp.weather,
            )

    def adapt_route_for_hazard(self, frame: FrameContext) -> PatrolWaypoint | None:
        """Suggest a lower-altitude re-inspection waypoint when hazards are suspected."""
        hint = str(frame.metadata.get("scene_hint", "")).lower()
        if not any(token in hint for token in ("smoke", "fire", "chemical", "spill", "crowd", "hazard")):
            return None
        return PatrolWaypoint(
            latitude=frame.telemetry.gps.latitude,
            longitude=frame.telemetry.gps.longitude,
            altitude_m=max(18.0, frame.telemetry.altitude_m * 0.55),
            scene_hint=f"close inspection: {hint}",
            lighting=frame.lighting,
            weather=frame.weather,
        )
