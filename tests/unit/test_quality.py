import unittest

import numpy as np

from src.face_auth.domain import reason_codes
from src.face_auth.domain.types import GateStatus
from src.face_auth.inference.quality import QualityConfig, QualityGate


class QualityGateTest(unittest.TestCase):
    def test_dark_flat_frame_is_retryable_quality_failure(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = QualityGate().evaluate(frame).result
        self.assertEqual(result.status, GateStatus.FAIL)
        self.assertIn(reason_codes.BLUR, result.reason_codes)
        self.assertIn(reason_codes.LOW_LIGHT, result.reason_codes)

    def test_textured_well_lit_frame_passes_relaxed_test_config(self):
        checker = np.indices((64, 64)).sum(axis=0) % 2 * 100 + 80
        frame = np.repeat(checker.astype(np.uint8)[..., None], 3, axis=2)
        gate = QualityGate(QualityConfig(min_blur_variance=1.0))
        self.assertEqual(gate.evaluate(frame).result.status, GateStatus.PASS)

    def test_invalid_bbox_is_rejected(self):
        with self.assertRaises(ValueError):
            QualityGate().evaluate(
                np.zeros((20, 20, 3), dtype=np.uint8), (10, 10, 5, 5)
            )


if __name__ == "__main__":
    unittest.main()
