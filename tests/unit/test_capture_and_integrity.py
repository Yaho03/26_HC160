import unittest

import numpy as np

from src.face_auth.adapters.capture_base import LatestFrameBuffer
from src.face_auth.domain.types import FramePacket, GateStatus
from src.face_auth.inference.frame_integrity import FrameIntegrityGate


def packet(frame_id, timestamp):
    return FramePacket(frame_id, timestamp, np.zeros((2, 2, 3), dtype=np.uint8))


class CaptureAndIntegrityTest(unittest.TestCase):
    def test_bounded_buffer_reports_drops_and_returns_latest(self):
        buffer = LatestFrameBuffer(max_frames=2)
        buffer.push(packet(1, 1.0))
        buffer.push(packet(2, 2.0))
        buffer.push(packet(3, 3.0))
        self.assertEqual(buffer.dropped_frames, 1)
        self.assertEqual(buffer.pop_latest().frame_id, 3)
        self.assertEqual(len(buffer), 0)

    def test_integrity_rejects_replayed_frame_id(self):
        gate = FrameIntegrityGate()
        self.assertEqual(gate.evaluate(packet(1, 1.0)).status, GateStatus.PASS)
        self.assertEqual(gate.evaluate(packet(1, 2.0)).status, GateStatus.FAIL)

    def test_integrity_rejects_non_monotonic_time(self):
        gate = FrameIntegrityGate()
        gate.evaluate(packet(1, 2.0))
        self.assertEqual(gate.evaluate(packet(2, 1.0)).status, GateStatus.FAIL)


if __name__ == "__main__":
    unittest.main()
