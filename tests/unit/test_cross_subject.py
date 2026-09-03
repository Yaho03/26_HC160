"""피험자 분리 검증.

07_DEFENSE_AND_DETECTION_SPEC.md 5절은 subject와 session 분리를 요구한다.
같은 피험자로 임계값을 정하고 같은 피험자로 평가하면 낙관 편향이 생긴다.
"""

import unittest

import numpy as np

from src.verification.defenses.cross_subject import (
    InsufficientSubjectsError,
    leave_one_subject_out,
    subject_splits,
)


def _rows(subject, session, label, values, transform="jpeg_q75"):
    out = []
    for index, value in enumerate(values):
        out.append({
            "subject_id": subject, "session_id": session,
            "sample_id": f"f{index:06d}_{label}", "label": label,
            "transform": transform,
            "cos_orig_enroll": 0.8,
            "cos_transformed_enroll": 0.8 - value,
            "cos_orig_transformed": 1.0 - value,
            "self_consistency": value, "template_shift": value,
        })
    return out


class SubjectSplitsTest(unittest.TestCase):
    def test_each_subject_is_held_out_once(self):
        splits = list(subject_splits(["p01", "p02", "p03"]))

        self.assertEqual(len(splits), 3)
        self.assertEqual({test for _, test in splits}, {"p01", "p02", "p03"})

    def test_training_set_excludes_the_held_out_subject(self):
        for train, test in subject_splits(["p01", "p02", "p03"]):
            with self.subTest(test=test):
                self.assertNotIn(test, train)
                self.assertEqual(len(train), 2)

    def test_two_subjects_still_produce_splits(self):
        splits = list(subject_splits(["p01", "p02"]))
        self.assertEqual(len(splits), 2)

    def test_single_subject_is_rejected(self):
        """한 명으로는 held-out 구성이 불가능하다."""
        with self.assertRaises(InsufficientSubjectsError):
            list(subject_splits(["p01"]))


class LeaveOneSubjectOutTest(unittest.TestCase):
    def _dataset(self):
        rows = []
        # p01은 값이 작고 p02는 크다. 임계값이 전이되지 않는 상황을 만든다.
        rows += _rows("p01", "s1", "clean", np.full(60, 0.01))
        rows += _rows("p01", "s1", "adversarial", np.full(20, 0.05))
        rows += _rows("p02", "s2", "clean", np.full(60, 0.10))
        rows += _rows("p02", "s2", "adversarial", np.full(20, 0.50))
        return rows

    def test_reports_one_result_per_held_out_subject(self):
        results = leave_one_subject_out(
            self._dataset(), features=[("jpeg_q75", "self_consistency")],
            target_fpr=0.05, window_frames=1,
        )
        self.assertEqual({r["test_subject"] for r in results}, {"p01", "p02"})

    def test_threshold_is_fitted_on_training_subjects_only(self):
        """테스트 피험자의 clean이 임계값에 들어가면 분리가 깨진다."""
        results = leave_one_subject_out(
            self._dataset(), features=[("jpeg_q75", "self_consistency")],
            target_fpr=0.05, window_frames=1,
        )
        by_subject = {r["test_subject"]: r for r in results}

        # p01 학습 → 낮은 값 기준이므로 p02 테스트에서는 임계값이 낮게 잡힌다
        self.assertNotEqual(
            by_subject["p01"]["threshold"], by_subject["p02"]["threshold"]
        )

    def test_reports_counts_with_the_rates(self):
        results = leave_one_subject_out(
            self._dataset(), features=[("jpeg_q75", "self_consistency")],
            target_fpr=0.05, window_frames=1,
        )
        for result in results:
            with self.subTest(subject=result["test_subject"]):
                self.assertIn("n_clean", result)
                self.assertIn("n_adversarial", result)
                self.assertIsNotNone(result["tpr"])


if __name__ == "__main__":
    unittest.main()
