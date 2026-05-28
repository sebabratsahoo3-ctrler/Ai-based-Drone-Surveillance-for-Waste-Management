from __future__ import annotations

from dataclasses import dataclass

from environmental_os.schemas import FrameContext


@dataclass(frozen=True)
class EdgeBudget:
    ram_mb: int = 4096
    max_inference_fps: float = 1.0
    patrol_frame_stride: int = 8
    hazard_frame_stride: int = 2
    low_power_frame_stride: int = 16
    base_resolution: tuple[int, int] = (640, 360)
    hazard_resolution: tuple[int, int] = (768, 432)
    tiny_resolution: tuple[int, int] = (384, 216)


class SparseFrameScheduler:
    def __init__(self, budget: EdgeBudget | None = None):
        self.budget = budget or EdgeBudget()
        self._frame_counts: dict[str, int] = {}

    def should_infer(self, frame: FrameContext) -> bool:
        drone_id = frame.telemetry.drone_id
        count = self._frame_counts.get(drone_id, 0) + 1
        self._frame_counts[drone_id] = count

        hint = " ".join([str(frame.metadata.get("scene_hint", "")), " ".join(frame.roi_hints)]).lower()
        if any(token in hint for token in ("smoke", "fire", "chemical", "spill", "panic", "flooded")):
            return count % self.budget.hazard_frame_stride == 0
        if frame.telemetry.battery_pct < 25:
            return count % self.budget.low_power_frame_stride == 0
        return count % self.budget.patrol_frame_stride == 0

    def target_resolution(self, frame: FrameContext) -> tuple[int, int]:
        hint = " ".join([str(frame.metadata.get("scene_hint", "")), " ".join(frame.roi_hints)]).lower()
        if any(token in hint for token in ("smoke", "fire", "chemical", "spill", "crowd", "vehicle")):
            return self.budget.hazard_resolution
        if frame.telemetry.battery_pct < 25:
            return self.budget.tiny_resolution
        return self.budget.base_resolution


def roi_policy(frame: FrameContext) -> list[str]:
    hints = list(frame.roi_hints)
    text = str(frame.metadata.get("scene_hint", "")).lower()
    if "drain" in text or "river" in text:
        hints.append("water-channel-roi")
    if "bin" in text or "market" in text:
        hints.append("collection-point-roi")
    if "vehicle" in text:
        hints.append("vehicle-edge-roi")
    if "smoke" in text or "fire" in text:
        hints.append("thermal-smoke-plume-roi")
    return sorted(set(hints))

