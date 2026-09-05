"""세션 내 robust z 정규화.

EXP-DET-001 18.2절. 전역 임계값을 쓰지 않고 각 평가 세션 안에서
중앙값 + k * 1.4826 * MAD 를 넘는 표본을 탐지로 본다. 세션 안에서는 조명이
일정하므로 등록 시점 통계로 미래 세션을 예측할 필요가 없다.

임계값이 평가 세션 자신에게서 나오므로 이 방식은 transductive하다. 그 성질과
한계는 문서에 적는다. 여기서는 계산이 정의대로인지만 고정한다.
"""

import unittest

import numpy as np

from src.verification.defenses.cross_subject import (
    MAD_TO_SIGMA,
    MIXED_ATTACK,
    MissingSessionTimeError,
    leave_one_subject_out,
)

TIMES = {
    "p01_e": "2026-09-01T08:00:00", "p01_t": "2026-09-02T08:00:00",
    "p02_e": "2026-09-01T09:00:00", "p02_t": "2026-09-02T09:00:00",
}


def _rows(subject, session, label, values, kind="", transform="jpeg_q75"):
    return [
        {
            "subject_id": subject, "session_id": session,
            # sample_id는 세션 안에서 고유해야 한다. 종류를 넣지 않으면 두 공격의
            # 표본이 같은 키로 겹쳐 하나로 접힌다.
            "sample_id": f"f{i:06d}_{label}{kind}", "label": label, "transform": transform,
            "attack_kind": kind if label != "clean" else "",
            "cos_orig_enroll": 0.8, "cos_transformed_enroll": 0.8 - v,
            "cos_orig_transformed": 1.0 - v,
            "self_consistency": v, "template_shift": v,
        }
        for i, v in enumerate(values)
    ]


FEATURES = [("jpeg_q75", "self_consistency")]


def _run(rows, **kwargs):
    options = dict(
        features=FEATURES, target_fpr=0.01, window_frames=1,
        normalization="session_robust", session_times=TIMES,
        min_session_clean=5,
    )
    options.update(kwargs)
    return leave_one_subject_out(rows, **options)


class ThresholdDefinitionTest(unittest.TestCase):
    """단일 특징이면 결합 점수가 곧 robust z이므로 임계값은 정확히 k가 된다."""

    def _dataset(self, adversarial_values, kind="pgd"):
        rows = []
        for subject in ("p01", "p02"):
            rows += _rows(subject, f"{subject}_e", "clean", [1.0, 2.0, 3.0, 4.0, 5.0])
            rows += _rows(subject, f"{subject}_t", "clean", [1.0, 2.0, 3.0, 4.0, 5.0])
            rows += _rows(
                subject, f"{subject}_t", "adversarial", adversarial_values, kind=kind
            )
        return rows

    def test_mad_constant_is_the_normal_consistency_factor(self):
        self.assertAlmostEqual(MAD_TO_SIGMA, 1.4826)

    def test_sample_just_above_the_cut_is_detected(self):
        """clean 중앙값 3, MAD 1. k=3이면 경계는 3 + 3 * 1.4826 * 1 = 7.4478이다."""
        results = _run(self._dataset([7.45, 7.45]), robust_k=3.0)

        for result in results:
            with self.subTest(subject=result["test_subject"]):
                self.assertEqual(result["tpr"], 1.0)

    def test_sample_just_below_the_cut_is_not_detected(self):
        results = _run(self._dataset([7.44, 7.44]), robust_k=3.0)

        for result in results:
            with self.subTest(subject=result["test_subject"]):
                self.assertEqual(result["tpr"], 0.0)

    def test_larger_k_raises_the_cut(self):
        detected_at_three = _run(self._dataset([7.45, 7.45]), robust_k=3.0)
        detected_at_five = _run(self._dataset([7.45, 7.45]), robust_k=5.0)

        self.assertEqual(detected_at_three[0]["tpr"], 1.0)
        self.assertEqual(detected_at_five[0]["tpr"], 0.0)

    def test_threshold_is_recorded_per_session_not_per_fold(self):
        results = _run(self._dataset([7.45, 7.45]), robust_k=3.0)

        for result in results:
            with self.subTest(subject=result["test_subject"]):
                self.assertIsNone(result["threshold"])
                self.assertEqual(result["threshold_source"], "session")
                self.assertEqual(len(result["thresholds_by_session"]), 1)


class ThresholdIndependenceTest(unittest.TestCase):
    """학습 피험자의 표본은 이 방식의 임계값에 들어가지 않는다."""

    def _dataset(self, other_scale):
        rows = []
        rows += _rows("p01", "p01_e", "clean", [1.0, 2.0, 3.0, 4.0, 5.0])
        rows += _rows("p01", "p01_t", "clean", [1.0, 2.0, 3.0, 4.0, 5.0])
        rows += _rows("p01", "p01_t", "adversarial", [7.45], kind="pgd")
        rows += _rows("p02", "p02_e", "clean", np.arange(1, 6) * other_scale)
        rows += _rows("p02", "p02_t", "clean", np.arange(1, 6) * other_scale)
        rows += _rows("p02", "p02_t", "adversarial", [7.45 * other_scale], kind="pgd")
        return rows

    def test_changing_training_subject_does_not_move_the_cut(self):
        small = _run(self._dataset(1.0), robust_k=3.0)
        large = _run(self._dataset(100.0), robust_k=3.0)

        cut_small = [r for r in small if r["test_subject"] == "p01"][0]
        cut_large = [r for r in large if r["test_subject"] == "p01"][0]
        self.assertEqual(
            cut_small["thresholds_by_session"], cut_large["thresholds_by_session"]
        )


class AttackBreakdownTest(unittest.TestCase):
    """집계 TPR만 내면 한 공격 종류의 완전 실패가 가려진다."""

    def _dataset(self):
        rows = []
        for subject in ("p01", "p02"):
            rows += _rows(subject, f"{subject}_e", "clean", [1.0, 2.0, 3.0, 4.0, 5.0])
            rows += _rows(subject, f"{subject}_t", "clean", [1.0, 2.0, 3.0, 4.0, 5.0])
            # 잡히는 공격 넷, 전혀 안 잡히는 공격 넷
            rows += _rows(subject, f"{subject}_t", "adversarial", [9.0] * 4, kind="pgd")
            rows += _rows(subject, f"{subject}_t", "adversarial", [3.1] * 4, kind="fgsm")
        return rows

    def test_aggregate_hides_a_fully_missed_attack_kind(self):
        result = _run(self._dataset(), robust_k=3.0)[0]

        self.assertEqual(result["tpr"], 0.5)
        self.assertEqual(result["tpr_by_attack"]["pgd"]["tpr"], 1.0)
        self.assertEqual(result["tpr_by_attack"]["fgsm"]["tpr"], 0.0)

    def test_breakdown_reports_counts_with_the_rate(self):
        result = _run(self._dataset(), robust_k=3.0)[0]

        self.assertEqual(result["tpr_by_attack"]["fgsm"]["n"], 4)
        self.assertEqual(result["tpr_by_attack"]["fgsm"]["detected"], 0)

    def test_breakdown_is_reported_for_every_normalization(self):
        """세션 내 robust z 에만 넣으면 다른 방식에서 같은 은폐가 재발한다."""
        for mode in ("population", "per_user", "session_robust"):
            with self.subTest(mode=mode):
                result = _run(self._dataset(), normalization=mode, robust_k=3.0)[0]
                self.assertIn("pgd", result["tpr_by_attack"])
                self.assertIn("fgsm", result["tpr_by_attack"])

    def test_windows_spanning_two_attack_kinds_are_labelled_mixed(self):
        """윈도가 종류 경계를 넘으면 그 윈도는 어느 종류로도 셀 수 없다."""
        result = _run(self._dataset(), window_frames=3, robust_k=3.0)[0]

        self.assertIn(MIXED_ATTACK, result["tpr_by_attack"])


class InsufficientSessionSamplesTest(unittest.TestCase):
    def _dataset(self, clean_values):
        rows = []
        for subject in ("p01", "p02"):
            rows += _rows(subject, f"{subject}_e", "clean", [1.0, 2.0, 3.0, 4.0, 5.0])
            rows += _rows(subject, f"{subject}_t", "clean", clean_values)
            rows += _rows(subject, f"{subject}_t", "adversarial", [9.0], kind="pgd")
        return rows

    def test_session_below_the_minimum_is_skipped_and_reported(self):
        result = _run(self._dataset([1.0, 2.0, 3.0]), robust_k=3.0)[0]

        self.assertEqual(len(result["skipped_sessions"]), 1)
        self.assertIsNone(result["tpr"])

    def test_zero_spread_session_is_skipped_rather_than_scaled_by_one(self):
        """MAD가 0인데 1로 대체하면 임계값이 임의의 단위로 정해진다."""
        result = _run(self._dataset([2.0] * 8), robust_k=3.0)[0]

        self.assertEqual(len(result["skipped_sessions"]), 1)
        self.assertIn("MAD", result["skipped_sessions"][0]["reason"])

    def test_skip_reason_names_the_session(self):
        result = _run(self._dataset([1.0, 2.0, 3.0]), robust_k=3.0)[0]

        self.assertIn("session", result["skipped_sessions"][0])


class SessionTimesRequiredTest(unittest.TestCase):
    def test_session_robust_needs_session_times(self):
        rows = _rows("p01", "p01_e", "clean", [1.0, 2.0]) + _rows(
            "p02", "p02_e", "clean", [1.0, 2.0]
        )
        with self.assertRaises(MissingSessionTimeError):
            leave_one_subject_out(
                rows, features=FEATURES, normalization="session_robust",
                window_frames=1,
            )


if __name__ == "__main__":
    unittest.main()
