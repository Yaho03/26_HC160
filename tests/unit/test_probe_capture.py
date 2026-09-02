import tempfile
import unittest
from pathlib import Path

from src.verification.defenses.probe_capture import (
    Preview,
    build_parser,
    decode_fourcc,
    sample_id,
    session_metadata,
    should_attack,
)
from src.verification.defenses.probe_log import write_session_sidecar
from src.verification.defenses.squeeze_probe import TRANSFORM_ORDER


class SampleIdTest(unittest.TestCase):
    def test_clean_and_adversarial_share_the_frame_prefix(self):
        clean = sample_id(7, "clean")
        adversarial = sample_id(7, "adversarial")

        self.assertTrue(clean.startswith("f000007_"))
        self.assertTrue(adversarial.startswith("f000007_"))
        self.assertNotEqual(clean, adversarial)


class ShouldAttackTest(unittest.TestCase):
    def test_fires_on_the_configured_period(self):
        fired = [index for index in range(30) if should_attack(index, 10)]
        self.assertEqual(fired, [0, 10, 20])

    def test_zero_disables_attack_generation(self):
        self.assertFalse(any(should_attack(index, 0) for index in range(30)))


class DecodeFourccTest(unittest.TestCase):
    """JPEG 변환의 해석이 코덱에 달려 있으므로 이 값을 반드시 남긴다."""

    def test_decodes_mjpg(self):
        packed = sum(ord(char) << (8 * shift) for shift, char in enumerate("MJPG"))
        self.assertEqual(decode_fourcc(packed), "MJPG")

    def test_decodes_raw_yuyv(self):
        packed = sum(ord(char) << (8 * shift) for shift, char in enumerate("YUY2"))
        self.assertEqual(decode_fourcc(packed), "YUY2")

    def test_returns_none_when_the_driver_reports_nothing(self):
        self.assertIsNone(decode_fourcc(0))
        self.assertIsNone(decode_fourcc(-1))


class PreviewTest(unittest.TestCase):
    """preview 실패가 계측 세션을 중단시키면 안 된다."""

    def test_disabled_preview_does_nothing(self):
        preview = Preview(enabled=False)
        preview.show(None, None, {}, 0, "")
        preview.close()

    def test_gui_failure_is_reported_once_then_suppressed(self):
        import cv2

        preview = Preview(enabled=True)
        calls = []

        def boom(*args, **kwargs):
            calls.append(1)
            raise cv2.error("no display")

        import src.verification.defenses.probe_capture as module

        original = module._draw
        module._draw = boom
        try:
            preview.show(None, None, {}, 0, "")
            preview.show(None, None, {}, 0, "")
        finally:
            module._draw = original

        self.assertEqual(len(calls), 1)
        self.assertTrue(preview._failed)


class ParserTest(unittest.TestCase):
    def test_preview_is_on_by_default(self):
        self.assertFalse(build_parser().parse_args(["--subject", "p01"]).no_preview)

    def test_no_preview_flag(self):
        self.assertTrue(
            build_parser().parse_args(["--subject", "p01", "--no-preview"]).no_preview
        )


class SessionMetadataTest(unittest.TestCase):
    def _metadata(self):
        return session_metadata(
            session_id="abc123",
            subject_id="p01",
            camera={
                "index": 0,
                "width": 1280,
                "height": 720,
                "fps_nominal": 30.0,
                "fourcc": "MJPG",
            },
            attack={"kind": "pgd_targeted_enroll", "epsilon": 0.03, "every": 10},
            counters={"frames_read": 300, "samples_clean": 300},
            elapsed_sec=20.0,
        )

    def test_partial_session_is_marked_incomplete(self):
        """중단된 세션도 사이드카를 남기되 완주 여부를 구분할 수 있어야 한다."""
        meta = session_metadata(
            session_id="abc123",
            subject_id="p01",
            camera={"index": 0, "fourcc": None},
            attack={"kind": "pgd_targeted_enroll"},
            counters={"frames_read": 127, "samples_clean": 106},
            elapsed_sec=120.0,
            target_frames=300,
            completed=False,
            interrupted_by="user_cancel",
        )

        self.assertFalse(meta["completed"])
        self.assertEqual(meta["interrupted_by"], "user_cancel")
        self.assertEqual(meta["target_frames"], 300)

    def test_completed_session_defaults_to_complete(self):
        self.assertTrue(self._metadata()["completed"])
        self.assertIsNone(self._metadata()["interrupted_by"])

    def test_records_every_transform_parameter(self):
        meta = self._metadata()
        self.assertEqual(tuple(meta["transforms"]), TRANSFORM_ORDER)

    def test_records_model_provenance_without_a_weights_path(self):
        model = self._metadata()["model"]

        self.assertEqual(model["pretrained"], "vggface2")
        self.assertIn("preprocess", model)
        self.assertNotIn("weights_path", model)

    def test_effective_fps_is_derived_from_elapsed_time(self):
        self.assertAlmostEqual(self._metadata()["effective_fps"], 15.0)

    def test_metadata_passes_the_sidecar_privacy_check(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "session.json"
            write_session_sidecar(target, self._metadata())
            self.assertTrue(target.exists())


if __name__ == "__main__":
    unittest.main()
