"""방어 전후 인증 결과 비교.

09_EVALUATION_METRICS.md 3절의 지표들은 분모가 다르므로 같은 label을 쓰면 안 된다.
07_DEFENSE_AND_DETECTION_SPEC.md 7절의 통과 기준은 conditional ASR 50% 이상 감소와
clean TAR 감소 2%p 이하다.
"""

import unittest

from src.verification.defenses.conditional_asr import (
    NoEligibleAttackError,
    clean_cost,
    conditional_defense_metrics,
)


class ConditionalDefenseMetricsTest(unittest.TestCase):
    def test_eligible_attempts_are_only_those_accepted_before_defense(self):
        """방어 전 거부된 공격은 분모에 들어가지 않는다."""
        result = conditional_defense_metrics(
            attack_similarity=[0.9, 0.9, 0.1],   # 마지막은 방어 전에 이미 거부
            attack_detected=[True, False, True],
            identity_threshold=0.5,
        )
        self.assertEqual(result["eligible"], 2)
        self.assertEqual(result["rejected_after_defense"], 1)

    def test_conditional_rates_use_the_eligible_denominator(self):
        result = conditional_defense_metrics(
            attack_similarity=[0.9] * 4,
            attack_detected=[True, True, True, False],
            identity_threshold=0.5,
        )
        self.assertAlmostEqual(result["conditional_defense_success_rate"], 0.75)
        self.assertAlmostEqual(result["conditional_asr_after_defense"], 0.25)

    def test_population_asr_uses_all_attempts_not_just_eligible(self):
        """conditional 과 population 은 분모가 다르다. 섞으면 안 된다."""
        result = conditional_defense_metrics(
            attack_similarity=[0.9, 0.9, 0.1, 0.1],
            attack_detected=[True, False, False, False],
            identity_threshold=0.5,
        )
        self.assertEqual(result["eligible"], 2)
        self.assertAlmostEqual(result["conditional_asr_after_defense"], 0.5)
        self.assertAlmostEqual(result["population_asr_after_defense"], 0.25)

    def test_refuses_when_no_attack_succeeded_before_defense(self):
        """분모가 0이면 0이 아니라 명시적 오류다."""
        with self.assertRaises(NoEligibleAttackError):
            conditional_defense_metrics(
                attack_similarity=[0.1, 0.2],
                attack_detected=[True, True],
                identity_threshold=0.5,
            )

    def test_reduction_against_no_defense_baseline(self):
        """방어 전 ASR은 정의상 1.0이다. 감소분은 곧 defense success rate다."""
        result = conditional_defense_metrics(
            attack_similarity=[0.9] * 10,
            attack_detected=[True] * 6 + [False] * 4,
            identity_threshold=0.5,
        )
        self.assertAlmostEqual(result["conditional_asr_before_defense"], 1.0)
        self.assertAlmostEqual(result["conditional_asr_reduction"], 0.6)
        self.assertTrue(result["meets_asr_budget"])

    def test_below_half_reduction_fails_the_budget(self):
        result = conditional_defense_metrics(
            attack_similarity=[0.9] * 10,
            attack_detected=[True] * 4 + [False] * 6,
            identity_threshold=0.5,
        )
        self.assertAlmostEqual(result["conditional_asr_reduction"], 0.4)
        self.assertFalse(result["meets_asr_budget"])


class CleanCostTest(unittest.TestCase):
    def test_clean_tar_delta_counts_only_newly_rejected_genuine_samples(self):
        """방어 전에 이미 거부된 clean은 방어 탓이 아니다."""
        result = clean_cost(
            clean_similarity=[0.9, 0.9, 0.9, 0.1],
            clean_detected=[False, True, False, True],
            identity_threshold=0.5,
        )
        self.assertEqual(result["accepted_before"], 3)
        self.assertEqual(result["accepted_after"], 2)
        self.assertAlmostEqual(result["clean_tar_before"], 0.75)
        self.assertAlmostEqual(result["clean_tar_after"], 0.50)
        self.assertAlmostEqual(result["clean_tar_delta_pp"], -25.0)

    def test_budget_is_two_percentage_points(self):
        result = clean_cost(
            clean_similarity=[0.9] * 100,
            clean_detected=[True] + [False] * 99,
            identity_threshold=0.5,
        )
        self.assertAlmostEqual(result["clean_tar_delta_pp"], -1.0)
        self.assertTrue(result["meets_clean_budget"])


if __name__ == "__main__":
    unittest.main()


class AttackModelScopeTest(unittest.TestCase):
    """통과 판정은 어떤 공격 모델에서 성립하는지와 함께 기록해야 한다.

    BPDA 평가에서 같은 방어가 conditional ASR 감소 0%를 기록했다. 공격 모델을
    적지 않으면 artifact가 조건 없는 주장을 하게 된다.
    """

    def test_attack_model_is_recorded(self):
        result = conditional_defense_metrics(
            attack_similarity=[0.9] * 4,
            attack_detected=[True, True, True, False],
            identity_threshold=0.5,
            attack_model="oblivious",
        )
        self.assertEqual(result["attack_model"], "oblivious")

    def test_budget_verdict_is_scoped_to_the_attack_model(self):
        oblivious = conditional_defense_metrics(
            attack_similarity=[0.9] * 10,
            attack_detected=[True] * 7 + [False] * 3,
            identity_threshold=0.5,
            attack_model="oblivious",
        )
        adaptive = conditional_defense_metrics(
            attack_similarity=[0.9] * 10,
            attack_detected=[False] * 10,
            identity_threshold=0.5,
            attack_model="bpda",
        )

        self.assertTrue(oblivious["meets_asr_budget"])
        self.assertFalse(adaptive["meets_asr_budget"])
        self.assertNotEqual(oblivious["attack_model"], adaptive["attack_model"])

    def test_unspecified_attack_model_defaults_to_unknown(self):
        """공격 모델을 모르면 모른다고 적는다. 비적응이라고 가정하지 않는다."""
        result = conditional_defense_metrics(
            attack_similarity=[0.9], attack_detected=[True], identity_threshold=0.5
        )
        self.assertEqual(result["attack_model"], "unspecified")
