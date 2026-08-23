from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

from src.face_auth.domain.types import FramePacket


class FrameSource(ABC):
    @abstractmethod
    def read(self) -> FramePacket | None:
        """Return the next frame or None at end of stream."""

    @abstractmethod
    def close(self) -> None:
        """Release camera or file resources."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class LatestFrameBuffer:
    """Bounded buffer that exposes overload instead of growing without limit."""

    def __init__(self, max_frames: int = 30) -> None:
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        self._frames: deque[FramePacket] = deque(maxlen=max_frames)
        self.dropped_frames = 0

    def push(self, frame: FramePacket) -> None:
        if len(self._frames) == self._frames.maxlen:
            self.dropped_frames += 1
        self._frames.append(frame)

    def pop_latest(self) -> FramePacket | None:
        if not self._frames:
            return None
        latest = self._frames.pop()
        self._frames.clear()
        return latest

    def snapshot(self) -> tuple[FramePacket, ...]:
        return tuple(self._frames)

    def __len__(self) -> int:
        return len(self._frames)
