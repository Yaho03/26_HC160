import unittest

from src.face_auth.evaluation.pad_metrics import (
    calibrate_pad_threshold,
    pad_metrics,
    reclassify_pad_samples,
)


def result(label, outcome, species="none"):
    return {"label": label, "outcome": outcome, "attack_species": species}


class PADMetricsTest(unittest.TestCase):
    def test_apcer_bpcer_exclude_errors_and_not_evaluated(self):
        metrics = pad_metrics(
            [
                result("bona_fide", "PASS"),
                result("bona_fide", "FAIL"),
                result("bona_fide", "ERROR"),
                result("attack", "PASS", "print"),
                result("attack", "FAIL", "print"),
                result("attack", "FAIL", "screen_replay"),
                result("attack", "NOT_EVALUATED", "screen_replay"),
            ]
        )
        self.assertEqual(metrics["bpcer"]["value"], 0.5)
        self.assertAlmostEqual(metrics["apcer"]["value"], 1 / 3)
        self.assertAlmostEqual(metrics["acer"], (0.5 + 1 / 3) / 2)
        self.assertEqual(metrics["sample_counts"]["error"], 1)
        self.assertEqual(metrics["sample_counts"]["not_evaluated"], 1)
        self.assertEqual(metrics["apcer_by_attack_species"]["print"]["denominator"], 2)
        self.assertEqual(
            metrics["apcer_by_attack_species"]["print"]["presentations"], 2
        )
        self.assertEqual(metrics["apcer_worst_species"]["attack_species"], "print")

    def test_undefined_rate_is_null_not_zero(self):
        metrics = pad_metrics([result("bona_fide", "PASS")])
        self.assertIsNone(metrics["apcer"]["value"])
        self.assertIsNone(metrics["acer"])

    def test_calibration_uses_bona_fide_rejection_budget(self):
        samples = [
            {**result("bona_fide", "PASS"), "score": 0.70},
            {**result("bona_fide", "PASS"), "score": 0.80},
            {**result("bona_fide", "PASS"), "score": 0.90},
            {**result("bona_fide", "PASS"), "score": 0.95},
            {**result("attack", "FAIL", "print"), "score": 0.20},
            {**result("attack", "FAIL", "screen_static"), "score": 0.75},
        ]
        candidate = calibrate_pad_threshold(samples, max_bpcer=0.25)
        self.assertEqual(candidate.threshold, 0.80)
        reclassified = reclassify_pad_samples(samples, candidate.threshold)
        metrics = pad_metrics(reclassified)
        self.assertEqual(metrics["bpcer"]["value"], 0.25)
        self.assertEqual(metrics["apcer"]["value"], 0.0)


if __name__ == "__main__":
    unittest.main()
