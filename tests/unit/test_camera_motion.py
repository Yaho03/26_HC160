import unittest

import cv2
import numpy as np

from src.face_auth.domain import reason_codes
from src.face_auth.domain.types import FramePacket, GateStatus
from src.face_auth.inference.camera_motion import CameraMotionConfig, CameraMotionGate


def packet(frame_id, image):
    return FramePacket(frame_id, float(frame_id + 1), image)


class CameraMotionGateTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        gray = rng.integers(0, 256, size=(120, 160), dtype=np.uint8)
        self.image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.gate = CameraMotionGate(
            CameraMotionConfig(
                max_normalized_motion=0.015,
                min_valid_pairs=2,
                min_tracked_points=8,
            )
        )

    def test_stable_camera_passes(self):
        frames = [packet(index, self.image.copy()) for index in range(3)]
        result = self.gate.evaluate(frames)
        self.assertEqual(result.status, GateStatus.PASS)

    def test_large_global_translation_is_retryable_camera_shake(self):
        matrix = np.float32([[1, 0, 12], [0, 1, 0]])
        shifted_once = cv2.warpAffine(self.image, matrix, (160, 120))
        shifted_twice = cv2.warpAffine(shifted_once, matrix, (160, 120))
        frames = [
            packet(0, self.image),
            packet(1, shifted_once),
            packet(2, shifted_twice),
        ]
        result = self.gate.evaluate(frames)
        self.assertEqual(result.status, GateStatus.FAIL)
        self.assertIn(reason_codes.CAMERA_SHAKE, result.reason_codes)


if __name__ == "__main__":
    unittest.main()
