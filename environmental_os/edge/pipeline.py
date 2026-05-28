from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from environmental_os.edge.scheduler import SparseFrameScheduler, roi_policy
from environmental_os.fusion.multimodal import SensorBundle
from environmental_os.schemas import EnvironmentalEvent, FrameContext
from environmental_os.vlm.models import VLMReasoner


@dataclass
class EdgeInferenceStats:
    frames_received: int = 0
    frames_skipped: int = 0
    inferences_run: int = 0
    total_processing_ms: float = 0.0
    cache_hits: int = 0

    @property
    def avg_processing_ms(self) -> float:
        if self.inferences_run == 0:
            return 0.0
        return self.total_processing_ms / self.inferences_run


@dataclass
class EdgeInferencePipeline:
    """Async edge pipeline: sparse scheduling, ROI crops, VLM inference, event emit."""

    vlm: VLMReasoner
    scheduler: SparseFrameScheduler = field(default_factory=SparseFrameScheduler)
    process_fn: Callable[[FrameContext], list[EnvironmentalEvent]] | None = None
    sensor_resolver: Callable[[FrameContext], SensorBundle] | None = None
    stats: EdgeInferenceStats = field(default_factory=EdgeInferenceStats)
    _queue: deque[FrameContext] = field(default_factory=deque)
    _summary_cache: dict[str, tuple[float, list]] = field(default_factory=dict)
    cache_ttl_s: float = 12.0

    async def submit(self, frame: FrameContext) -> list[EnvironmentalEvent]:
        self.stats.frames_received += 1
        frame.roi_hints = roi_policy(frame)
        resolution = self.scheduler.target_resolution(frame)
        frame.resolution = resolution

        if not self.scheduler.should_infer(frame):
            self.stats.frames_skipped += 1
            return []

        sensors = self.sensor_resolver(frame) if self.sensor_resolver else None
        cache_key = self._cache_key(frame, sensors=sensors)
        cached = self._summary_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < self.cache_ttl_s:
            self.stats.cache_hits += 1
            return cached[1]

        events = await self._run_inference(frame, sensors=sensors)
        self._summary_cache[cache_key] = (time.time(), events)
        return events

    async def _run_inference(self, frame: FrameContext, sensors: SensorBundle | None = None) -> list[EnvironmentalEvent]:
        started = time.perf_counter()
        if self.process_fn:
            try:
                events = await asyncio.to_thread(self.process_fn, frame, sensors)
            except TypeError:
                events = await asyncio.to_thread(self.process_fn, frame)
        else:
            events = []
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.stats.inferences_run += 1
        self.stats.total_processing_ms += elapsed_ms
        return events

    async def drain_queue(self) -> list[EnvironmentalEvent]:
        all_events: list[EnvironmentalEvent] = []
        while self._queue:
            frame = self._queue.popleft()
            all_events.extend(await self.submit(frame))
        return all_events

    def enqueue(self, frame: FrameContext) -> None:
        self._queue.append(frame)

    @staticmethod
    def _cache_key(frame: FrameContext, sensors: SensorBundle | None = None) -> str:
        zone = frame.telemetry.gps.rounded_zone()
        hint = str(frame.metadata.get("scene_hint", ""))[:80]
        if not sensors:
            return f"{frame.telemetry.drone_id}:{zone}:{hint}:{frame.lighting}"
        # Sensor streams (audio/thermal/weather/crowd) can materially change
        # the interpretation, so include a compact signature in the cache key.
        sensor_sig = ",".join(
            [
                f"a{round(sensors.audio_anomaly_score, 2)}",
                f"t{round(sensors.thermal_hotspot_score, 2)}",
                f"w{round(sensors.weather_risk, 2)}",
                f"c{round(sensors.crowd_density, 2)}",
            ]
        )
        return f"{frame.telemetry.drone_id}:{zone}:{hint}:{frame.lighting}:{sensor_sig}"
