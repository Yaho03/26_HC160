from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.face_auth.domain import reason_codes
from src.face_auth.domain.types import FramePacket, GateResult, GateStatus


@dataclass(frozen=True)
class CameraMotionConfig:
    max_normalized_motion: float = 0.035
    min_valid_pairs: int = 2
    min_tracked_points: int = 12
    threshold_version: str = "camera-motion-v1"


class CameraMotionGate:
    """Measures global frame translation with sparse optical flow.

    Motion is normalized by the frame diagonal so one threshold can be used at
    different resolutions. Face regions are masked when bboxes are available.
    """

    def __init__(self, config: CameraMotionConfig | None = None) -> None:
        self.config = config or CameraMotionConfig()

    def evaluate(
        self,
        frames: list[FramePacket],
        face_bboxes: dict[int, tuple[int, int, int, int]] | None = None,
    ) -> GateResult:
        pair_scores: list[float] = []
        for previous, current in zip(frames, frames[1:]):
            score = self._pair_motion(previous, current, face_bboxes or {})
            if score is not None:
                pair_scores.append(score)

        if len(pair_scores) < self.config.min_valid_pairs:
            return GateResult(
                "camera_motion",
                GateStatus.FAIL,
                reason_codes=(reason_codes.INSUFFICIENT_VALID_FRAMES,),
                threshold=self.config.max_normalized_motion,
                threshold_version=self.config.threshold_version,
            )

        score = float(np.percentile(np.asarray(pair_scores), 90))
        passed = score <= self.config.max_normalized_motion
        return GateResult(
            "camera_motion",
            GateStatus.PASS if passed else GateStatus.FAIL,
            score=score,
            threshold=self.config.max_normalized_motion,
            reason_codes=() if passed else (reason_codes.CAMERA_SHAKE,),
            threshold_version=self.config.threshold_version,
        )

    def _pair_motion(
        self,
        previous: FramePacket,
        current: FramePacket,
        bboxes: dict[int, tuple[int, int, int, int]],
    ) -> float | None:
        previous_gray = _gray(previous.image_bgr)
        current_gray = _gray(current.image_bgr)
        if previous_gray.shape != current_gray.shape:
            return None

        mask = np.full(previous_gray.shape, 255, dtype=np.uint8)
        bbox = bboxes.get(previous.frame_id)
        if bbox is not None:
            _mask_expanded_bbox(mask, bbox)

        points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=7,
            mask=mask,
        )
        if points is None or len(points) < self.config.min_tracked_points:
            return None

        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            points,
            None,
        )
        if moved is None or status is None:
            return None
        valid = status.reshape(-1).astype(bool)
        if int(valid.sum()) < self.config.min_tracked_points:
            return None

        displacement = np.linalg.norm(
            moved.reshape(-1, 2)[valid] - points.reshape(-1, 2)[valid],
            axis=1,
        )
        diagonal = float(np.hypot(*previous_gray.shape))
        return float(np.median(displacement) / diagonal)


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _mask_expanded_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    height, width = mask.shape
    x1, y1, x2, y2 = bbox
    pad_x = int(max(0, x2 - x1) * 0.25)
    pad_y = int(max(0, y2 - y1) * 0.25)
    x1, x2 = max(0, x1 - pad_x), min(width, x2 + pad_x)
    y1, y2 = max(0, y1 - pad_y), min(height, y2 + pad_y)
    mask[y1:y2, x1:x2] = 0
