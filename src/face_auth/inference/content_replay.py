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
        if len(frames) < 2:
            return GateResult(
                "content_replay",
                GateStatus.FAIL,
                reason_codes=(reason_codes.INSUFFICIENT_VALID_FRAMES,),
                threshold=float(self.config.max_near_duplicate_run),
                threshold_version=self.config.threshold_version,
            )
        fingerprints = [
            _fingerprint(frame.image_bgr, self.config.fingerprint_size)
            for frame in frames
        ]
        longest_run = 0
        current_run = 0
        for previous, current in zip(fingerprints, fingerprints[1:]):
            difference = float(np.mean(np.abs(current - previous)))
            if difference <= self.config.max_mean_absolute_difference:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 0
        passed = longest_run <= self.config.max_near_duplicate_run
        return GateResult(
            "content_replay",
            GateStatus.PASS if passed else GateStatus.FAIL,
            score=float(longest_run),
            threshold=float(self.config.max_near_duplicate_run),
            reason_codes=() if passed else (reason_codes.FRAME_SEQUENCE_INVALID,),
            threshold_version=self.config.threshold_version,
        )


def _fingerprint(frame: np.ndarray, size: int) -> np.ndarray:
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )
