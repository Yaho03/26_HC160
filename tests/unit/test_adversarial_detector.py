import unittest

import numpy as np

from src.face_auth.domain import reason_codes
from src.face_auth.domain.types import GateStatus
from src.face_auth.inference.adversarial_detector import (
    AdversarialDetectorConfig,
    TransformConsistencyDetector,
)


class TransformConsistencyDetectorTest(unittest.TestCase):
    def setUp(self):
        self.detector = TransformConsistencyDetector(
            AdversarialDetectorConfig(0.1, "adv-dispersion-v1")
        )

    def test_stable_transforms_pass(self):
        original = np.array([1.0, 0.0], dtype=np.float32)
        transformed = [
            np.array([1.0, 0.01], dtype=np.float32),
            np.array([1.0, -0.01], dtype=np.float32),
        ]
        self.assertEqual(
            self.detector.evaluate(original, transformed).status,
            GateStatus.PASS,
        )

    def test_unstable_transform_fails(self):
        original = np.array([1.0, 0.0], dtype=np.float32)
        transformed = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
        result = self.detector.evaluate(original, transformed)
        self.assertEqual(result.status, GateStatus.FAIL)
        self.assertIn(reason_codes.ADVERSARIAL_SUSPECTED, result.reason_codes)


if __name__ == "__main__":
    unittest.main()
