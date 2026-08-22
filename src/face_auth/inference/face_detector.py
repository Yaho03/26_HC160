from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from src.face_auth.domain import reason_codes
from src.face_auth.domain.types import GateResult, GateStatus


@dataclass(frozen=True)
class DetectedFace:
    bbox: tuple[int, int, int, int]
    probability: float
    crop: Image.Image
    landmarks: tuple[tuple[float, float], ...] | None = None


class MTCNNFaceDetector:
    """Baseline detector; all faces are returned and no face is silently selected."""

    def __init__(self, *, image_size: int = 160, margin: int = 20, device=None) -> None:
        from facenet_pytorch import MTCNN

        self.image_size = image_size
        self._detector = MTCNN(
            image_size=image_size,
            margin=margin,
            keep_all=True,
            device=device,
        )

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        boxes, probabilities, landmarks = self._detector.detect(image, landmarks=True)
        if boxes is None:
            return []
        detected: list[DetectedFace] = []
        for index, (box, probability) in enumerate(zip(boxes, probabilities)):
            bbox = tuple(int(round(value)) for value in box)
            crop = _crop_face(image, bbox, self.image_size)
            face_landmarks = None
            if landmarks is not None:
                face_landmarks = tuple(
                    (float(point[0]), float(point[1])) for point in landmarks[index]
                )
            detected.append(
                DetectedFace(
                    bbox=bbox,
                    probability=float(probability or 0.0),
                    crop=crop,
                    landmarks=face_landmarks,
                )
            )
        return detected


def face_count_gate(faces: list[DetectedFace]) -> GateResult:
    if len(faces) == 1:
        return GateResult(gate="single_face", status=GateStatus.PASS, score=1.0)
    if not faces:
        return GateResult(
            gate="single_face",
            status=GateStatus.FAIL,
            score=0.0,
            reason_codes=(reason_codes.NO_FACE,),
        )
    return GateResult(
        gate="single_face",
        status=GateStatus.FAIL,
        score=float(len(faces)),
        reason_codes=(reason_codes.MULTIPLE_FACES,),
    )


def _crop_face(
    image: Image.Image, bbox: tuple[int, int, int, int], size: int
) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid face bbox: {bbox}")
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Face bbox is outside the image: {bbox}")
    return image.crop((x1, y1, x2, y2)).resize((size, size), Image.BILINEAR)
