from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from environmental_os.schemas import FrameContext


@dataclass
class FrameBuffer:
    """Ring buffer for recent drone frames used in temporal VLM reasoning."""

    max_frames: int = 24
    frames: deque[FrameContext] = field(default_factory=deque)

    def push(self, frame: FrameContext) -> None:
        self.frames.append(frame)
        while len(self.frames) > self.max_frames:
            self.frames.popleft()

    def recent(self, count: int = 8) -> list[FrameContext]:
        return list(self.frames)[-count:]

    def latest(self) -> FrameContext | None:
        return self.frames[-1] if self.frames else None

    def clear(self) -> None:
        self.frames.clear()
