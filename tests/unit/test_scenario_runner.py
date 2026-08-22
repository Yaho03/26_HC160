import unittest

import numpy as np

from src.attack_scenarios.scenario_runner import InsertFrames, RepeatFrame, apply_events
from src.face_auth.domain.types import FramePacket


def packet(frame_id):
    return FramePacket(
        frame_id, float(frame_id + 1), np.zeros((1, 1, 3), dtype=np.uint8)
    )


class ScenarioRunnerTest(unittest.TestCase):
    def test_inserts_attack_frames_at_requested_index(self):
        result = apply_events(
            [packet(0), packet(1), packet(2)],
            [InsertFrames(1, (packet(100), packet(101)))],
        )
        self.assertEqual([frame.frame_id for frame in result], [0, 100, 101, 1, 2])

    def test_repeat_preserves_metadata_for_replay_detection(self):
        result = apply_events(
            [packet(0), packet(1), packet(2)],
            [RepeatFrame(source_index=0, at_index=2, count=2)],
        )
        self.assertEqual([frame.frame_id for frame in result], [0, 1, 0, 0, 2])


if __name__ == "__main__":
    unittest.main()
