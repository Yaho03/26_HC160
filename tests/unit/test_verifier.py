import unittest

import numpy as np

from src.face_auth.domain import reason_codes
from src.face_auth.domain.types import GateStatus
from src.face_auth.inference.verifier import (
    MultiFrameVerifier,
    VerificationConfig,
    build_template,
)


class MultiFrameVerifierTest(unittest.TestCase):
    def setUp(self):
        enrollment = [
            np.array([1.0, offset], dtype=np.float32)
            for offset in (0.0, 0.01, -0.01, 0.02, -0.02)
        ]
        template = build_template(enrollment)
        self.verifier = MultiFrameVerifier(
            template,
            VerificationConfig(
                threshold=0.8,
                threshold_version="test-threshold-v1",
                model_version="fake-model-v1",
            ),
        )

    def test_similar_multi_frame_probe_passes(self):
        probes = [
            np.array([1.0, value], dtype=np.float32) for value in (0.01, 0.02, -0.01)
        ]
        self.assertEqual(self.verifier.evaluate(probes).status, GateStatus.PASS)

    def test_different_probe_fails(self):
        probes = [np.array([0.0, 1.0], dtype=np.float32) for _ in range(3)]
        result = self.verifier.evaluate(probes)
        self.assertEqual(result.status, GateStatus.FAIL)
        self.assertIn(reason_codes.LOW_IDENTITY_SIMILARITY, result.reason_codes)

    def test_insufficient_probe_frames_is_retryable_failure(self):
        result = self.verifier.evaluate([np.array([1.0, 0.0], dtype=np.float32)])
        self.assertEqual(result.status, GateStatus.FAIL)
        self.assertIn(reason_codes.INSUFFICIENT_VALID_FRAMES, result.reason_codes)


if __name__ == "__main__":
    unittest.main()
