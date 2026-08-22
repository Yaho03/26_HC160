import unittest

import numpy as np

from src.face_auth.domain import reason_codes
from src.face_auth.domain.types import GateStatus
from src.face_auth.inference.continuity import IdentityContinuityGate


class IdentityContinuityGateTest(unittest.TestCase):
    def setUp(self):
        self.gate = IdentityContinuityGate(np.array([1.0, 0.0], dtype=np.float32))

    def test_consistent_window_passes(self):
        embeddings = [np.array([1.0, 0.05], dtype=np.float32) for _ in range(5)]
        self.assertEqual(self.gate.evaluate(embeddings).status, GateStatus.PASS)

    def test_three_of_five_identity_switch_fails(self):
        embeddings = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
        result = self.gate.evaluate(embeddings)
        self.assertEqual(result.status, GateStatus.FAIL)
        self.assertIn(reason_codes.IDENTITY_SWITCH, result.reason_codes)


if __name__ == "__main__":
    unittest.main()
