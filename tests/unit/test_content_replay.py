import unittest

import numpy as np

from src.face_auth.domain import reason_codes
from src.face_auth.domain.types import FramePacket, GateStatus
from src.face_auth.inference.content_replay import ContentReplayGate


def packet(frame_id, value):
    image = np.full((64, 64, 3), value, dtype=np.uint8)
    return FramePacket(frame_id, float(frame_id + 1), image)


class ContentReplayGateTest(unittest.TestCase):
    def test_changing_frames_pass(self):
        result = ContentReplayGate().evaluate(
            [packet(index, 20 + index) for index in range(5)]
        )
        self.assertEqual(result.status, GateStatus.PASS)

    def test_long_frozen_run_is_rejected(self):
        result = ContentReplayGate().evaluate([packet(index, 20) for index in range(5)])
        self.assertEqual(result.status, GateStatus.FAIL)
        self.assertIn(reason_codes.FRAME_SEQUENCE_INVALID, result.reason_codes)


if __name__ == "__main__":
    unittest.main()
