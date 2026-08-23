import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.face_auth.adapters.opencv_capture import (
    CaptureSourceError,
    OpenCVCaptureSource,
)
from src.face_auth.adapters.opencv_preview import OpenCVPreview
from src.face_auth.cli import (
    CaptureBatch,
    ChallengeBindingError,
    _challenge_instruction,
    _collect,
    _preview_enabled,
    _resolve_challenge_start_frame_id,
    build_parser,
)
from src.face_auth.domain.types import FramePacket
from src.face_auth.inference.content_replay import ContentReplayMonitor


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

    def test_collect_binds_challenge_to_first_displayed_frame(self):
        source = FakeSource([packet(7), packet(8), packet(9)])
        preview = FakePreview([True, True, True])

        result = _collect(
            source,
            3,
            preview=preview,
            purpose="AUTHENTICATION",
            instruction="TURN HEAD LEFT",
        )

        self.assertEqual(result.challenge_start_frame_id, 7)
        self.assertEqual(
            preview.calls[0][1]["instruction"], "TURN HEAD LEFT"
        )

    def test_headless_collect_does_not_invent_displayed_challenge_boundary(self):
        source = FakeSource([packet(0), packet(1), packet(2)])

        result = _collect(
            source,
            3,
            instruction="TURN HEAD LEFT",
        )

        self.assertIsNone(result.challenge_start_frame_id)

    def test_live_replay_veto_stops_capture_on_first_threshold_violation(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        source = FakeSource(
            [FramePacket(index, float(index), image.copy()) for index in range(10)]
        )
        preview = FakePreview([True] * 10)

        result = _collect(
            source,
            10,
            preview=preview,
            purpose="AUTHENTICATION",
            instruction="TURN HEAD LEFT",
            live_replay_monitor=ContentReplayMonitor(),
        )

        self.assertEqual(len(result.frames), 4)
        self.assertIsNotNone(result.live_veto)
        self.assertEqual(result.live_veto.status.value, "FAIL")
        self.assertEqual(preview.calls[-1][1]["alert"], "REPLAY DETECTED")
        self.assertEqual(preview.calls[-1][1]["wait_ms"], 900)

    def test_live_replay_monitor_allows_changing_frames_to_finish(self):
        frames = []
        for index in range(6):
            image = np.full((240, 320, 3), index + 20, dtype=np.uint8)
            frames.append(FramePacket(index, float(index), image))
        source = FakeSource(frames)
        preview = FakePreview([True] * len(frames))

        result = _collect(
            source,
            len(frames),
            preview=preview,
            instruction="TURN HEAD RIGHT",
            live_replay_monitor=ContentReplayMonitor(),
        )

        self.assertEqual(len(result.frames), len(frames))
        self.assertIsNone(result.live_veto)

    def test_headless_replay_monitor_starts_at_external_challenge_marker(self):
        frames = []
        for index in range(8):
            image = np.full((240, 320, 3), 20, dtype=np.uint8)
            frames.append(FramePacket(index, float(index), image))
        source = FakeSource(frames)

        result = _collect(
            source,
            len(frames),
            live_replay_monitor=ContentReplayMonitor(),
            monitor_start_frame_id=2,
        )

        self.assertEqual(len(result.frames), 6)
        self.assertIsNotNone(result.live_veto)

    def test_headless_full_requires_explicit_valid_challenge_boundary(self):
        capture = CaptureBatch(tuple(packet(index) for index in range(6)))
        args = SimpleNamespace(challenge_start_frame_id=None, min_valid_frames=3)
        with self.assertRaises(ChallengeBindingError):
            _resolve_challenge_start_frame_id(args, capture)

        args.challenge_start_frame_id = 1
        self.assertEqual(_resolve_challenge_start_frame_id(args, capture), 1)

        args.challenge_start_frame_id = 4
        with self.assertRaisesRegex(
            ChallengeBindingError, "Not enough post-challenge frames"
        ):
            _resolve_challenge_start_frame_id(args, capture)

    def test_preview_and_external_boundaries_cannot_be_mixed(self):
        capture = CaptureBatch(
            tuple(packet(index) for index in range(6)),
            challenge_start_frame_id=0,
        )
        args = SimpleNamespace(challenge_start_frame_id=2, min_valid_frames=3)

        with self.assertRaisesRegex(ChallengeBindingError, "Do not provide"):
            _resolve_challenge_start_frame_id(args, capture)

    def test_challenge_kinds_have_stable_user_instructions(self):
        self.assertEqual(_challenge_instruction("HEAD_LEFT"), "TURN HEAD LEFT")
        self.assertEqual(_challenge_instruction("HEAD_RIGHT"), "TURN HEAD RIGHT")
        self.assertEqual(_challenge_instruction("BLINK"), "BLINK ONCE")
        with self.assertRaises(ChallengeBindingError):
            _challenge_instruction("SMILE")

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
        auth_args = parser.parse_args(
            [
                "authenticate",
                "--video",
                "probe.mp4",
                "--template",
                "template.npz",
                "--threshold",
                "0.7",
                "--threshold-version",
                "identity-v1",
                "--user-id",
                "user-1",
                "--decision-output",
                "decision.json",
            ]
        )

        self.assertTrue(_preview_enabled(camera_args))
        self.assertFalse(_preview_enabled(headless_args))
        self.assertFalse(_preview_enabled(video_args))
        self.assertEqual(auth_args.decision_output, "decision.json")
        self.assertFalse(auth_args.overwrite_decision_output)

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
                instruction="TURN HEAD LEFT",
            )
            rendered = imshow.call_args.args[1]
            preview.close()

        self.assertFalse(should_continue)
        self.assertTrue(np.any(rendered != 0))

    def test_preview_uses_requested_hold_for_security_alert(self):
        preview = OpenCVPreview("test-window")
        with (
            patch("src.face_auth.adapters.opencv_preview.cv2.imshow") as imshow,
            patch(
                "src.face_auth.adapters.opencv_preview.cv2.waitKey",
                return_value=-1,
            ) as wait_key,
            patch("src.face_auth.adapters.opencv_preview.cv2.destroyWindow"),
        ):
            should_continue = preview.show(
                packet(0),
                captured_frames=4,
                target_frames=20,
                purpose="AUTHENTICATION",
                instruction="TURN HEAD RIGHT",
                alert="REPLAY DETECTED",
                wait_ms=900,
            )
            rendered = imshow.call_args.args[1]
            preview.close()

        self.assertTrue(should_continue)
        wait_key.assert_any_call(900)
        self.assertGreater(int(rendered[-20, 10, 2]), 150)


if __name__ == "__main__":
    unittest.main()
