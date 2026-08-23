import unittest

import numpy as np

from src.face_auth.domain import reason_codes
from src.face_auth.domain.types import FramePacket, GateStatus
from src.face_auth.inference.content_replay import (
    ContentReplayConfig,
    ContentReplayGate,
    ContentReplayMonitor,
)


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
        frames = [packet(index, 20) for index in range(5)]
        result = ContentReplayGate().evaluate(frames)
        self.assertEqual(result.status, GateStatus.FAIL)
        self.assertIn(reason_codes.FRAME_SEQUENCE_INVALID, result.reason_codes)

    def test_incremental_monitor_vetoes_on_first_excess_duplicate(self):
        monitor = ContentReplayMonitor()
        statuses = [
            monitor.update(packet(index, 20)).status for index in range(5)
        ]

        self.assertEqual(
            statuses[:3],
            [GateStatus.NOT_EVALUATED, GateStatus.PASS, GateStatus.PASS],
        )
        self.assertEqual(statuses[3:], [GateStatus.FAIL, GateStatus.FAIL])
        self.assertEqual(monitor.longest_run, 4)

    def test_incremental_and_batch_results_match(self):
        frames = [packet(0, 20), packet(1, 20), packet(2, 21), packet(3, 21)]
        monitor = ContentReplayMonitor()
        for item in frames:
            monitor.update(item)

        streaming = monitor.result(require_pair=True)
        batch = ContentReplayGate().evaluate(frames)

        self.assertEqual(streaming.status, batch.status)
        self.assertEqual(streaming.score, batch.score)
        self.assertEqual(streaming.threshold, batch.threshold)
        self.assertEqual(streaming.threshold_version, batch.threshold_version)

    def test_invalid_monitor_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            ContentReplayMonitor(
                config=ContentReplayConfig(max_near_duplicate_run=-1)
            )


if __name__ == "__main__":
    unittest.main()
