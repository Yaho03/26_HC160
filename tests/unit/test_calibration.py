import unittest

from src.face_auth.evaluation.calibration import calibrate_threshold


class ThresholdCalibrationTest(unittest.TestCase):
    def test_higher_is_clean_threshold_uses_validation_frr_budget(self):
        result = calibrate_threshold(
            [0.70, 0.80, 0.90, 0.95],
            [0.20, 0.40, 0.75],
            higher_is_clean=True,
            max_clean_frr=0.25,
        )
        self.assertEqual(result.threshold, 0.80)
        self.assertEqual(result.clean_false_reject_rate, 0.25)
        self.assertAlmostEqual(result.attack_false_accept_rate, 0.0)

    def test_lower_is_clean_threshold_uses_validation_frr_budget(self):
        result = calibrate_threshold(
            [0.01, 0.02, 0.03, 0.04],
            [0.03, 0.08, 0.12],
            higher_is_clean=False,
            max_clean_frr=0.25,
        )
        self.assertEqual(result.threshold, 0.03)
        self.assertEqual(result.clean_false_reject_rate, 0.25)
        self.assertAlmostEqual(result.attack_false_accept_rate, 1 / 3)


if __name__ == "__main__":
    unittest.main()
