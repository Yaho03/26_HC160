from __future__ import annotations

import cv2
import numpy as np

from src.face_auth.domain.types import FramePacket


class PreviewUnavailableError(RuntimeError):
    """Raised when OpenCV cannot create or update a preview window."""


class OpenCVPreview:
    """Minimal camera preview that never persists captured frames."""

    _CANCEL_KEYS = {27, ord("q"), ord("Q")}

    def __init__(self, window_name: str = "HC160 Face Authentication") -> None:
        self.window_name = window_name
        self._opened = False

    def show(
        self,
        packet: FramePacket,
        *,
        captured_frames: int,
        target_frames: int,
        purpose: str,
        instruction: str | None = None,
    ) -> bool:
        rendered = self._render(
            packet.image_bgr,
            captured_frames=captured_frames,
            target_frames=target_frames,
            purpose=purpose,
            instruction=instruction,
        )
        try:
            self._opened = True
            cv2.imshow(self.window_name, rendered)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as error:
            raise PreviewUnavailableError(
                "Cannot open the camera preview window. Use --no-preview only "
                "for an intentional headless run."
            ) from error
        return key not in self._CANCEL_KEYS

    def close(self) -> None:
        if not self._opened:
            return
        try:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass
        finally:
            self._opened = False

    @staticmethod
    def _render(
        frame_bgr: np.ndarray,
        *,
        captured_frames: int,
        target_frames: int,
        purpose: str,
        instruction: str | None = None,
    ) -> np.ndarray:
        rendered = frame_bgr.copy()
        height, width = rendered.shape[:2]
        banner_height = (
            min(128, max(96, height // 4))
            if instruction
            else min(96, max(64, height // 5))
        )

        overlay = rendered.copy()
        cv2.rectangle(overlay, (0, 0), (width, banner_height), (18, 24, 32), -1)
        cv2.addWeighted(overlay, 0.82, rendered, 0.18, 0, rendered)

        margin_x = max(12, width // 5)
        guide_top = banner_height + max(12, height // 20)
        guide_bottom = max(guide_top + 1, height - max(20, height // 12))
        cv2.rectangle(
            rendered,
            (margin_x, guide_top),
            (width - margin_x, guide_bottom),
            (96, 214, 148),
            2,
        )

        progress = min(1.0, captured_frames / max(1, target_frames))
        bar_left, bar_right = 18, max(19, width - 18)
        bar_top, bar_bottom = banner_height - 13, banner_height - 7
        cv2.rectangle(
            rendered,
            (bar_left, bar_top),
            (bar_right, bar_bottom),
            (78, 88, 101),
            -1,
        )
        cv2.rectangle(
            rendered,
            (bar_left, bar_top),
            (bar_left + round((bar_right - bar_left) * progress), bar_bottom),
            (96, 214, 148),
            -1,
        )

        cv2.putText(
            rendered,
            purpose.upper(),
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if instruction:
            cv2.putText(
                rendered,
                instruction,
                (18, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (96, 214, 148),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            rendered,
            f"CAPTURING {captured_frames}/{target_frames}",
            (18, banner_height - 24 if instruction else 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (205, 215, 225),
            1,
            cv2.LINE_AA,
        )
        return rendered
