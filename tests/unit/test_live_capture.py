import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.face_auth.adapters.opencv_capture import (
    CaptureSourceError,
    OpenCVCaptureSource,
)
from src.face_auth.adapters.opencv_preview import OpenCVPreview
from src.face_auth.cli import _collect, _preview_enabled, build_parser
from src.face_auth.domain.types import FramePacket


def packet(frame_id: int) -> FramePacket:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    return FramePacket(frame_id, float(frame_id), image)


class FakeSource:
    def __init__(self, frames):
        self.frames = list(frames)
        self.closed = False

    def read(self):
        return self.frames.pop(0) if self.frames else None

    def close(self):
        self.closed = True


class FakePreview:
    def __init__(self, responses):
        self.responses = list(responses)
        self.closed = False
        self.calls = []

    def show(self, frame, **kwargs):
        self.calls.append((frame, kwargs))
        return self.responses.pop(0)

    def close(self):
        self.closed = True


class LiveCaptureTest(unittest.TestCase):
    def test_unavailable_camera_has_stable_reason_code_and_releases_device(self):
        capture = Mock()
        capture.isOpened.return_value = False
        with patch(
            "src.face_auth.adapters.opencv_capture.cv2.VideoCapture",
            return_value=capture,
        ):
            with self.assertRaises(CaptureSourceError) as caught:
                OpenCVCaptureSource(3)

        self.assertEqual(caught.exception.reason_code, "CAMERA_UNAVAILABLE")
        capture.release.assert_called_once_with()

    def test_unavailable_video_has_distinct_reason_code(self):
        capture = Mock()
        capture.isOpened.return_value = False
        with patch(
            "src.face_auth.adapters.opencv_capture.cv2.VideoCapture",
            return_value=capture,
        ):
            with self.assertRaises(CaptureSourceError) as caught:
                OpenCVCaptureSource("missing.mp4")

        self.assertEqual(caught.exception.reason_code, "VIDEO_UNAVAILABLE")

    def test_collect_closes_source_and_preview(self):
        source = FakeSource([packet(0), packet(1)])
        preview = FakePreview([True, True])

        result = _collect(source, 2, preview=preview, purpose="ENROLLMENT")

        self.assertFalse(result.cancelled)
        self.assertEqual(len(result.frames), 2)
        self.assertTrue(source.closed)
        self.assertTrue(preview.closed)
        self.assertEqual(preview.calls[-1][1]["captured_frames"], 2)

    def test_collect_returns_cancelled_and_releases_resources(self):
        source = FakeSource([packet(0), packet(1), packet(2)])
        preview = FakePreview([True, False])

        result = _collect(source, 3, preview=preview, purpose="AUTHENTICATION")

        self.assertTrue(result.cancelled)
        self.assertEqual(len(result.frames), 2)
        self.assertTrue(source.closed)
        self.assertTrue(preview.closed)

    def test_camera_defaults_to_preview_and_can_be_disabled(self):
        parser = build_parser()
        camera_args = parser.parse_args(
            ["enroll", "--camera", "0", "--output", "template.npz"]
        )
        headless_args = parser.parse_args(
            [
                "enroll",
                "--camera",
                "0",
                "--no-preview",
                "--output",
                "template.npz",
            ]
        )
        video_args = parser.parse_args(
            ["enroll", "--video", "probe.mp4", "--output", "template.npz"]
        )

        self.assertTrue(_preview_enabled(camera_args))
        self.assertFalse(_preview_enabled(headless_args))
        self.assertFalse(_preview_enabled(video_args))

    def test_preview_renders_overlay_and_q_cancels(self):
        preview = OpenCVPreview("test-window")
        with (
            patch("src.face_auth.adapters.opencv_preview.cv2.imshow") as imshow,
            patch(
                "src.face_auth.adapters.opencv_preview.cv2.waitKey",
                return_value=ord("q"),
            ),
            patch("src.face_auth.adapters.opencv_preview.cv2.destroyWindow"),
        ):
            should_continue = preview.show(
                packet(0),
                captured_frames=1,
                target_frames=20,
                purpose="AUTHENTICATION",
            )
            rendered = imshow.call_args.args[1]
            preview.close()

        self.assertFalse(should_continue)
        self.assertTrue(np.any(rendered != 0))


if __name__ == "__main__":
    unittest.main()
