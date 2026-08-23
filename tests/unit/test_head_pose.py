import unittest

from PIL import Image

from src.face_auth.inference.face_detector import DetectedFace
from src.face_auth.inference.head_pose import FivePointHeadPoseEstimator


class FivePointHeadPoseEstimatorTest(unittest.TestCase):
    def test_nose_offset_produces_right_turn_yaw_proxy(self):
        face = DetectedFace(
            (0, 0, 64, 64),
            1.0,
            Image.new("RGB", (64, 64)),
            ((20, 20), (44, 20), (54, 32), (24, 48), (40, 48)),
        )
        sample = FivePointHeadPoseEstimator().estimate(12, face)
        self.assertGreater(sample.yaw_degrees, 15.0)
        self.assertEqual(sample.frame_id, 12)

    def test_missing_landmarks_is_not_silently_accepted(self):
        face = DetectedFace((0, 0, 64, 64), 1.0, Image.new("RGB", (64, 64)))
        with self.assertRaises(ValueError):
            FivePointHeadPoseEstimator().estimate(1, face)


if __name__ == "__main__":
    unittest.main()
