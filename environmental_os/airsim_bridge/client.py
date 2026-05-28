from __future__ import annotations

from datetime import datetime, timezone
from itertools import count

from environmental_os.schemas import DroneTelemetry, FrameContext, GPSPoint


class AirSimDroneBridge:
    """Thin optional AirSim client wrapper.

    The project can run without AirSim installed. When the `airsim` package and
    simulator are available, replace `mock_frame` with calls to image, GPS, and
    vehicle state APIs.
    """

    def __init__(self, drone_id: str = "drone-001"):
        self.drone_id = drone_id
        self._counter = count(1)
        self._client = None

    def connect(self) -> bool:
        try:
            import airsim  # type: ignore
        except ImportError:
            return False
        self._client = airsim.MultirotorClient()
        self._client.confirmConnection()
        self._client.enableApiControl(True)
        self._client.armDisarm(True)
        return True

    def mock_frame(
        self,
        scene_hint: str,
        gps: GPSPoint | None = None,
        altitude_m: float = 42.0,
        lighting: str = "day",
    ) -> FrameContext:
        idx = next(self._counter)
        gps = gps or GPSPoint(latitude=12.9716, longitude=77.5946)
        telemetry = DroneTelemetry(
            drone_id=self.drone_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            gps=gps,
            altitude_m=altitude_m,
            speed_mps=4.2,
            heading_deg=85.0,
            battery_pct=72.0,
            mission_id="airsim-sanitation-patrol",
        )
        return FrameContext(
            frame_id=f"{self.drone_id}-frame-{idx:05d}",
            frame_ref=f"airsim://{self.drone_id}/frame/{idx:05d}",
            telemetry=telemetry,
            lighting=lighting,
            metadata={"scene_hint": scene_hint},
        )

