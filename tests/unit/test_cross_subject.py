"""피험자 분리 검증.

07_DEFENSE_AND_DETECTION_SPEC.md 5절은 subject와 session 분리를 요구한다.
같은 피험자로 임계값을 정하고 같은 피험자로 평가하면 낙관 편향이 생긴다.
"""

import unittest

import numpy as np

from src.verification.defenses.cross_subject import (
    InsufficientSubjectsError,
    MissingSessionTimeError,
    NoEnrollmentSessionError,
    enrollment_split,
    leave_one_subject_out,
    order_sessions,
    session_times_from_sidecars,
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


class OrderSessionsTest(unittest.TestCase):
    """등록 세션은 촬영 시각으로 정한다. 세션 ID는 무작위라 순서 정보가 없다."""

    def test_orders_by_capture_time_not_by_identifier(self):
        times = {"zzz": "2026-09-05T07:12:57", "aaa": "2026-09-05T07:35:13"}

        self.assertEqual(order_sessions(["aaa", "zzz"], times), ["zzz", "aaa"])

    def test_missing_time_is_rejected_instead_of_falling_back(self):
        """조용히 ID 정렬로 돌아가면 의도와 구현이 어긋난 채로 지나간다."""
        with self.assertRaises(MissingSessionTimeError):
            order_sessions(["aaa", "zzz"], {"aaa": "2026-09-05T07:35:13"})

    def test_none_time_is_rejected(self):
        with self.assertRaises(MissingSessionTimeError):
            order_sessions(["aaa"], {"aaa": None})

    def test_ties_break_on_session_id_deterministically(self):
        """같은 시각이면 순서를 정의해 둔다. 실행마다 달라지면 재현이 깨진다."""
        times = {"bbb": "2026-09-05T07:00:00", "aaa": "2026-09-05T07:00:00"}

        self.assertEqual(order_sessions(["bbb", "aaa"], times), ["aaa", "bbb"])
        self.assertEqual(order_sessions(["aaa", "bbb"], times), ["aaa", "bbb"])


class SessionTimesFromSidecarsTest(unittest.TestCase):
    def test_reads_created_at_by_session_id(self):
        times = session_times_from_sidecars(
            [{"session_id": "s1", "created_at": "2026-09-05T07:00:00"}]
        )

        self.assertEqual(times, {"s1": "2026-09-05T07:00:00"})

    def test_sidecar_without_created_at_is_rejected(self):
        with self.assertRaises(MissingSessionTimeError):
            session_times_from_sidecars([{"session_id": "s1"}])


class EnrollmentSplitTest(unittest.TestCase):
    TIMES = {"late": "2026-09-05T07:35:13", "early": "2026-09-05T07:12:57"}

    def _rows_two_sessions(self):
        rows = _rows("p01", "late", "clean", np.full(4, 0.02))
        rows += _rows("p01", "early", "clean", np.full(4, 0.01))
        return rows

    def test_earliest_assignment_uses_the_first_capture(self):
        enrollment, test = enrollment_split(
            self._rows_two_sessions(), "p01",
            session_times=self.TIMES, enrollment="earliest",
        )

        self.assertEqual({r["session_id"] for r in enrollment}, {"early"})
        self.assertEqual({r["session_id"] for r in test}, {"late"})

    def test_latest_assignment_uses_the_last_capture(self):
        """등록 배정 민감도를 재려면 반대 배정도 돌릴 수 있어야 한다."""
        enrollment, test = enrollment_split(
            self._rows_two_sessions(), "p01",
            session_times=self.TIMES, enrollment="latest",
        )

        self.assertEqual({r["session_id"] for r in enrollment}, {"late"})
        self.assertEqual({r["session_id"] for r in test}, {"early"})

    def test_unknown_assignment_is_rejected(self):
        with self.assertRaises(ValueError):
            enrollment_split(
                self._rows_two_sessions(), "p01",
                session_times=self.TIMES, enrollment="middle",
            )

    def test_single_session_subject_is_rejected(self):
        rows = _rows("p01", "early", "clean", np.full(4, 0.01))
        with self.assertRaises(NoEnrollmentSessionError):
            enrollment_split(rows, "p01", session_times=self.TIMES)


class PerUserRequiresSessionTimesTest(unittest.TestCase):
    def _dataset(self):
        rows = []
        for subject, base in (("p01", 0.01), ("p02", 0.10)):
            for index, session in enumerate(("s1_" + subject, "s2_" + subject)):
                rows += _rows(subject, session, "clean", np.full(30, base + index * 0.001))
                rows += _rows(subject, session, "adversarial", np.full(10, base * 5))
        return rows

    def test_per_user_without_session_times_is_rejected(self):
        """시각 없이 돌면 다시 ID 정렬로 조용히 돌아간다. 그것을 막는다."""
        with self.assertRaises(MissingSessionTimeError):
            leave_one_subject_out(
                self._dataset(), features=[("jpeg_q75", "self_consistency")],
                target_fpr=0.05, window_frames=1, normalization="per_user",
            )

    def test_population_does_not_need_session_times(self):
        """전역 정규화는 등록 분할을 쓰지 않으므로 시각이 필요 없다."""
        results = leave_one_subject_out(
            self._dataset(), features=[("jpeg_q75", "self_consistency")],
            target_fpr=0.05, window_frames=1, normalization="population",
        )

        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
