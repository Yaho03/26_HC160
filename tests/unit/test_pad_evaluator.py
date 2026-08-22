import unittest

import numpy as np
from PIL import Image

from src.face_auth.domain import reason_codes
from src.face_auth.domain.types import FramePacket
from src.face_auth.evaluation.pad_evaluator import PADVideoEvaluator
from src.face_auth.evaluation.pad_manifest import PADManifestRow
from src.face_auth.inference.face_detector import DetectedFace
from src.face_auth.inference.passive_pad import PassivePADConfig, PassivePADGate
from src.face_auth.inference.quality import QualityConfig, QualityGate


class FakeDetector:
    def __init__(self, count=1):
        self.count = count

    def detect(self, frame_bgr):
        crop = Image.fromarray(frame_bgr[:, :, ::-1])
        return [DetectedFace((0, 0, 64, 64), 1.0, crop) for _ in range(self.count)]


class FakeScorer:
    def __init__(self, score):
        self.value = score

    def score(self, crops):
        return [self.value for _ in crops]


def manifest_row(label="bona_fide", species="none"):
    return PADManifestRow(
        "sample_00000001",
        label,
        species,
        "sample.mp4",
        "subject_00000001",
        "session_00000001",
        "device_00000001",
        "test",
    )


def frames(count=5):
    rng = np.random.default_rng(3)
    image = rng.integers(40, 210, size=(64, 64, 3), dtype=np.uint8)
    return [
        FramePacket(index, float(index + 1), image.copy()) for index in range(count)
    ]


def evaluator(score, detector=None):
    return PADVideoEvaluator(
        detector or FakeDetector(),
        FakeScorer(score),
        PassivePADGate(PassivePADConfig(0.8, "pad-v1", "threshold-v1")),
        quality=QualityGate(QualityConfig(min_blur_variance=1.0)),
    )


class PADVideoEvaluatorTest(unittest.TestCase):
    def test_live_score_above_threshold_passes(self):
        result = evaluator(0.95).evaluate_frames(manifest_row(), frames())
        self.assertEqual(result.outcome, "PASS")
        self.assertEqual(result.valid_face_frames, 5)

    def test_spoof_score_below_threshold_fails(self):
        result = evaluator(0.2).evaluate_frames(
            manifest_row("attack", "print"), frames()
        )
        self.assertEqual(result.outcome, "FAIL")
        self.assertIn(reason_codes.SPOOF_SUSPECTED, result.reason_codes)

    def test_multiple_faces_are_not_counted_as_pad_rejection(self):
        result = evaluator(0.2, FakeDetector(count=2)).evaluate_frames(
            manifest_row("attack", "print"), frames()
        )
        self.assertEqual(result.outcome, "NOT_EVALUATED")
        self.assertIn(reason_codes.MULTIPLE_FACES, result.reason_codes)

    def test_too_few_frames_are_not_counted_as_pad_rejection(self):
        result = evaluator(0.2).evaluate_frames(
            manifest_row("attack", "print"), frames(2)
        )
        self.assertEqual(result.outcome, "NOT_EVALUATED")


if __name__ == "__main__":
    unittest.main()
