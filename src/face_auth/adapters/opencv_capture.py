from __future__ import annotations

import time

import cv2

from src.face_auth.adapters.capture_base import FrameSource
from src.face_auth.domain.types import FramePacket


class CaptureSourceError(RuntimeError):
    def __init__(self, source: int | str) -> None:
        self.source = source
        self.reason_code = (
            "CAMERA_UNAVAILABLE" if isinstance(source, int) else "VIDEO_UNAVAILABLE"
        )
        source_kind = "camera" if isinstance(source, int) else "video"
        super().__init__(f"Cannot open {source_kind} capture source: {source}")


class OpenCVCaptureSource(FrameSource):
    def __init__(self, source: int | str, *, backend: int | None = None) -> None:
        self.source = source
        self._capture = (
            cv2.VideoCapture(source)
            if backend is None
            else cv2.VideoCapture(source, backend)
        )
        if not self._capture.isOpened():
            self._capture.release()
            raise CaptureSourceError(source)
        self._next_frame_id = 0

    def read(self) -> FramePacket | None:
        ok, frame = self._capture.read()
        if not ok:
            return None
        packet = FramePacket(
            frame_id=self._next_frame_id,
            captured_at_monotonic=time.monotonic(),
            source_time_ms=float(self._capture.get(cv2.CAP_PROP_POS_MSEC)),
            image_bgr=frame,
        )
        self._next_frame_id += 1
        return packet

    def close(self) -> None:
        self._capture.release()


def webcam_source(camera_index: int = 0) -> OpenCVCaptureSource:
    """Use OpenCV's platform default instead of a macOS-only backend."""
    return OpenCVCaptureSource(camera_index)


def video_source(path: str) -> OpenCVCaptureSource:
    return OpenCVCaptureSource(path)
