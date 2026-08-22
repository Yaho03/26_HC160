from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.face_auth.domain import reason_codes
from src.face_auth.domain.types import GateResult, GateStatus


@dataclass(frozen=True)
class QualityConfig:
    min_blur_variance: float = 40.0
    min_mean_brightness: float = 35.0
    max_mean_brightness: float = 220.0


@dataclass(frozen=True)
class QualityAssessment:
    result: GateResult
    blur_variance: float
    mean_brightness: float


class QualityGate:
    def __init__(self, config: QualityConfig | None = None) -> None:
        self.config = config or QualityConfig()

    def evaluate(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> QualityAssessment:
        roi = _safe_roi(frame_bgr, bbox)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())

        reasons: list[str] = []
        if blur < self.config.min_blur_variance:
            reasons.append(reason_codes.BLUR)
        if brightness < self.config.min_mean_brightness:
            reasons.append(reason_codes.LOW_LIGHT)
        elif brightness > self.config.max_mean_brightness:
            reasons.append(reason_codes.OVEREXPOSED)

        result = GateResult(
            gate="quality",
            status=GateStatus.FAIL if reasons else GateStatus.PASS,
            score=blur,
            threshold=self.config.min_blur_variance,
            reason_codes=tuple(reasons),
            threshold_version="quality-v1",
        )
        return QualityAssessment(
            result=result, blur_variance=blur, mean_brightness=brightness
        )


def _safe_roi(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
) -> np.ndarray:
    if frame.size == 0:
        raise ValueError("Empty frame")
    if bbox is None:
        return frame
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid face bbox: {bbox}")
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid face bbox: {bbox}")
    return frame[y1:y2, x1:x2]
