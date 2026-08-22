from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.face_auth.inference.active_liveness import LivenessSample
from src.face_auth.inference.face_detector import DetectedFace


@dataclass(frozen=True)
class FivePointHeadPoseConfig:
    yaw_scale_degrees: float = 45.0


class FivePointHeadPoseEstimator:
    """Produces a calibrated yaw proxy from MTCNN's five landmarks.

    Five landmarks cannot provide a reliable blink estimate, so `eyes_closed`
    remains false. The prototype challenge issuer uses head turns for this
    adapter; blink requires a dense-landmark implementation.
    """

    def __init__(self, config: FivePointHeadPoseConfig | None = None) -> None:
        self.config = config or FivePointHeadPoseConfig()

    def estimate(self, frame_id: int, face: DetectedFace) -> LivenessSample:
        if face.landmarks is None or len(face.landmarks) != 5:
            raise ValueError("Five face landmarks are required for head pose")
        points = np.asarray(face.landmarks, dtype=np.float32)
        left_eye, right_eye, nose = points[0], points[1], points[2]
        interocular = float(np.linalg.norm(right_eye - left_eye))
        if interocular <= 1e-6:
            raise ValueError("Invalid eye landmarks")
        eye_mid_x = float((left_eye[0] + right_eye[0]) / 2.0)
        normalized_offset = float((nose[0] - eye_mid_x) / interocular)
        yaw = normalized_offset * self.config.yaw_scale_degrees
        return LivenessSample(frame_id=frame_id, yaw_degrees=yaw, eyes_closed=False)
