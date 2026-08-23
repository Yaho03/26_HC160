from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.face_auth.domain import reason_codes
from src.face_auth.domain.types import FramePacket, GateResult, GateStatus


@dataclass(frozen=True)
class ContentReplayConfig:
    max_near_duplicate_run: int = 2
    max_mean_absolute_difference: float = 0.30
    fingerprint_size: int = 32
    threshold_version: str = "content-replay-v2"


class ContentReplayGate:
    """Rejects camera freezes and consecutive frame-content replay."""

    def __init__(self, config: ContentReplayConfig | None = None) -> None:
        self.config = config or ContentReplayConfig()

    def evaluate(self, frames: list[FramePacket]) -> GateResult:
        monitor = ContentReplayMonitor(self.config)
        for frame in frames:
            monitor.update(frame)
        return monitor.result(require_pair=True)


class ContentReplayMonitor:
    """Incremental replay detector with the same contract as the batch gate."""

    def __init__(self, config: ContentReplayConfig | None = None) -> None:
        self.config = config or ContentReplayConfig()
        if self.config.max_near_duplicate_run < 0:
            raise ValueError("max_near_duplicate_run must be non-negative")
        if self.config.max_mean_absolute_difference < 0:
            raise ValueError("max_mean_absolute_difference must be non-negative")
        if self.config.fingerprint_size < 1:
            raise ValueError("fingerprint_size must be positive")
        self._previous: np.ndarray | None = None
        self.current_run = 0
        self.longest_run = 0
        self.observed_frames = 0

    def update(self, frame: FramePacket) -> GateResult:
        current = _fingerprint(frame.image_bgr, self.config.fingerprint_size)
        self.observed_frames += 1
        if self._previous is not None:
            difference = float(np.mean(np.abs(current - self._previous)))
            if difference <= self.config.max_mean_absolute_difference:
                self.current_run += 1
                self.longest_run = max(self.longest_run, self.current_run)
            else:
                self.current_run = 0
        self._previous = current
        return self.result()

    def result(self, *, require_pair: bool = False) -> GateResult:
        if self.observed_frames < 2:
            return GateResult(
                "content_replay",
                GateStatus.FAIL if require_pair else GateStatus.NOT_EVALUATED,
                score=float(self.longest_run),
                threshold=float(self.config.max_near_duplicate_run),
                reason_codes=(reason_codes.INSUFFICIENT_VALID_FRAMES,),
                threshold_version=self.config.threshold_version,
            )
        passed = self.longest_run <= self.config.max_near_duplicate_run
        return GateResult(
            "content_replay",
            GateStatus.PASS if passed else GateStatus.FAIL,
            score=float(self.longest_run),
            threshold=float(self.config.max_near_duplicate_run),
            reason_codes=() if passed else (reason_codes.FRAME_SEQUENCE_INVALID,),
            threshold_version=self.config.threshold_version,
        )


def _fingerprint(frame: np.ndarray, size: int) -> np.ndarray:
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )
