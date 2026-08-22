import unittest

from src.face_auth.domain.types import GateStatus
from src.face_auth.inference.active_liveness import ActiveLivenessGate, LivenessSample


def sample(frame_id, yaw=0.0, closed=False):
    return LivenessSample(frame_id, yaw, closed)


class ActiveLivenessGateTest(unittest.TestCase):
    def setUp(self):
        self.gate = ActiveLivenessGate()

    def test_head_turn_after_challenge_passes(self):
        result = self.gate.evaluate(
            "HEAD_LEFT",
            [sample(1, -20), sample(11, 0), sample(12, -18), sample(13, -20)],
            challenge_start_frame_id=10,
        )
        self.assertEqual(result.status, GateStatus.PASS)

    def test_action_only_before_challenge_does_not_pass(self):
        result = self.gate.evaluate(
            "HEAD_LEFT",
            [sample(1, -20), sample(11, 0), sample(12, -2), sample(13, -3)],
            challenge_start_frame_id=10,
        )
        self.assertEqual(result.status, GateStatus.FAIL)

    def test_open_closed_open_blink_passes(self):
        result = self.gate.evaluate(
            "BLINK",
            [sample(11), sample(12, closed=True), sample(13)],
            challenge_start_frame_id=10,
        )
        self.assertEqual(result.status, GateStatus.PASS)


if __name__ == "__main__":
    unittest.main()
