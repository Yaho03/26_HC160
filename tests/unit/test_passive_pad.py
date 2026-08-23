import math
import unittest

from src.face_auth.domain import reason_codes
from src.face_auth.domain.types import GateStatus
from src.face_auth.inference.passive_pad import PassivePADConfig, PassivePADGate


class PassivePADGateTest(unittest.TestCase):
    def setUp(self):
        self.gate = PassivePADGate(PassivePADConfig(0.7, "fake-pad-v1", "pad-th-v1"))

    def test_live_window_passes(self):
        self.assertEqual(self.gate.evaluate([0.8] * 5).status, GateStatus.PASS)

    def test_spoof_window_fails(self):
        result = self.gate.evaluate([0.2] * 5)
        self.assertEqual(result.status, GateStatus.FAIL)
        self.assertIn(reason_codes.SPOOF_SUSPECTED, result.reason_codes)

    def test_non_finite_model_output_is_error(self):
        result = self.gate.evaluate([0.8, 0.8, math.nan, 0.8, 0.8])
        self.assertEqual(result.status, GateStatus.ERROR)


if __name__ == "__main__":
    unittest.main()
