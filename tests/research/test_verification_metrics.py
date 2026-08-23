import unittest

from src.evaluation.verification_metrics import (
    attack_transition_summary,
    defense_transition_summary,
    verification_rates,
    wilson_interval,
)


class VerificationMetricsTest(unittest.TestCase):
    def test_verification_confusion_rates(self):
        rates = verification_rates(
            same_identity=[True, True, False, False],
            accepted=[True, False, True, False],
        )
        self.assertEqual(rates.counts.true_positive, 1)
        self.assertEqual(rates.counts.false_negative, 1)
        self.assertEqual(rates.counts.false_positive, 1)
        self.assertEqual(rates.counts.true_negative, 1)
        self.assertEqual(rates.far.value, 0.5)
        self.assertEqual(rates.frr.value, 0.5)

    def test_zero_denominator_is_undefined_not_zero(self):
        rates = verification_rates([True, True], [True, False])
        self.assertIsNone(rates.far.value)
        self.assertEqual(rates.far.denominator, 0)

    def test_targeted_attack_counts_only_reject_to_accept(self):
        summary = attack_transition_summary(
            accepted_before=[False, False, True, True],
            accepted_after=[True, False, True, False],
        )
        self.assertEqual(summary.eligible_reject_before, 2)
        self.assertEqual(summary.reject_to_accept, 1)
        self.assertEqual(summary.already_accepted_before, 2)
        self.assertEqual(summary.targeted_asr.value, 0.5)

    def test_defense_rates_expose_conditional_and_population_denominators(self):
        # Mirrors the committed smoothing report shape: 212 total, 171 attacks
        # accepted before defense, 167 blocked, and 4 still accepted.
        before = [True] * 171 + [False] * 41
        after = [False] * 167 + [True] * 4 + [False] * 41
        summary = defense_transition_summary(before, after)
        self.assertEqual(summary.blocked_after_defense, 167)
        self.assertEqual(summary.conditional_defense_success_rate.denominator, 171)
        self.assertAlmostEqual(summary.conditional_defense_success_rate.value, 167 / 171)
        self.assertEqual(summary.conditional_asr_after_defense.denominator, 171)
        self.assertAlmostEqual(summary.conditional_asr_after_defense.value, 4 / 171)
        self.assertEqual(summary.population_asr_after_defense.denominator, 212)
        self.assertAlmostEqual(summary.population_asr_after_defense.value, 4 / 212)

    def test_wilson_interval_contains_observed_rate(self):
        interval = wilson_interval(80, 100)
        self.assertIsNotNone(interval)
        self.assertLess(interval[0], 0.8)
        self.assertGreater(interval[1], 0.8)


if __name__ == "__main__":
    unittest.main()
