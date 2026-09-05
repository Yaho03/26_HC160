"""사용자별 정규화.

임계값이 사람 사이에 전이되지 않는 원인은 사람마다 점수 분포가 다르기 때문이다.
얼굴 인증에는 등록 절차가 있으므로 등록 시점에 그 사람의 clean 분포를 측정해
정규화할 수 있다. 등록 통계는 배포 환경에서 실제로 쓸 수 있는 정보이므로 누수가
아니다.

다만 등록 세션과 테스트 세션은 분리해야 한다. 같은 세션으로 정규화하고 평가하면
피험자 내부 누수가 된다.
"""

import unittest

import numpy as np

from src.verification.defenses.cross_subject import (
    NoEnrollmentSessionError,
    enrollment_split,
    leave_one_subject_out,
)


def _rows(subject, session, label, values, transform="jpeg_q75"):
    return [
        {
            "subject_id": subject, "session_id": session,
            "sample_id": f"f{i:06d}_{label}", "label": label, "transform": transform,
            "cos_orig_enroll": 0.8, "cos_transformed_enroll": 0.8 - v,
            "cos_orig_transformed": 1.0 - v,
            "self_consistency": v, "template_shift": v,
        }
        for i, v in enumerate(values)
    ]


# 세션 ID 정렬과 촬영 순서가 어긋나게 둔다. 실제 데이터가 그랬다.
TIMES = {"s1": "2026-09-05T09:00:00", "s2": "2026-09-05T08:00:00",
         "sa": "2026-09-05T09:00:00", "sb": "2026-09-05T08:00:00"}


class EnrollmentSplitTest(unittest.TestCase):
    def test_earliest_capture_is_enrollment_not_lowest_identifier(self):
        """세션 ID 정렬은 촬영 순서가 아니다. s2가 먼저 찍혔으므로 등록이다."""
        rows = _rows("p01", "s2", "clean", [0.1]) + _rows("p01", "s1", "clean", [0.2])
        enrollment, test = enrollment_split(rows, "p01", session_times=TIMES)

        self.assertEqual({r["session_id"] for r in enrollment}, {"s2"})
        self.assertEqual({r["session_id"] for r in test}, {"s1"})

    def test_single_session_subject_cannot_be_split(self):
        """등록과 테스트가 같은 세션이면 피험자 내부 누수가 된다."""
        rows = _rows("p01", "s1", "clean", [0.1, 0.2])
        with self.assertRaises(NoEnrollmentSessionError):
            enrollment_split(rows, "p01", session_times=TIMES)

    def test_split_is_deterministic(self):
        rows = _rows("p01", "sb", "clean", [0.1]) + _rows("p01", "sa", "clean", [0.2])
        first = enrollment_split(rows, "p01", session_times=TIMES)[0][0]["session_id"]
        second = enrollment_split(
            list(reversed(rows)), "p01", session_times=TIMES
        )[0][0]["session_id"]
        self.assertEqual(first, second)


class PerUserNormalizationTest(unittest.TestCase):
    def _dataset(self):
        """두 피험자의 점수 규모가 10배 다르지만 분리 구조는 같다."""
        rng = np.random.default_rng(0)
        rows = []
        for subject, scale in (("p01", 1.0), ("p02", 10.0)):
            for session in ("s1", "s2"):
                rows += _rows(subject, f"{subject}_{session}", "clean",
                              rng.normal(1.0, 0.1, 60) * scale)
                rows += _rows(subject, f"{subject}_{session}", "adversarial",
                              rng.normal(3.0, 0.1, 20) * scale)
        return rows

    def _times(self):
        return {
            f"{subject}_{session}": f"2026-09-0{index + 1}T08:00:00"
            for subject in ("p01", "p02")
            for index, session in enumerate(("s1", "s2"))
        }

    def _run(self, normalization, enrollment="earliest"):
        return leave_one_subject_out(
            self._dataset(), features=[("jpeg_q75", "self_consistency")],
            target_fpr=0.05, window_frames=1, normalization=normalization,
            session_times=self._times(), enrollment=enrollment,
        )

    def test_population_normalization_fails_to_transfer(self):
        """규모가 다르면 전역 통계로는 임계값이 맞지 않는다."""
        results = {r["test_subject"]: r for r in self._run("population")}
        self.assertLess(min(r["tpr"] for r in results.values()), 0.5)

    def test_per_user_normalization_transfers(self):
        """자기 등록 통계로 정규화하면 규모 차이가 사라진다."""
        results = {r["test_subject"]: r for r in self._run("per_user")}
        for subject, result in results.items():
            with self.subTest(subject=subject):
                self.assertGreater(result["tpr"], 0.9)

    def test_normalization_mode_is_reported(self):
        for mode in ("population", "per_user"):
            with self.subTest(mode=mode):
                self.assertTrue(all(r["normalization"] == mode for r in self._run(mode)))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self._run("wishful")

    def test_both_enrollment_assignments_run(self):
        """등록 배정 민감도를 재려면 두 배정 모두 돌아가야 한다."""
        for assignment in ("earliest", "latest"):
            with self.subTest(assignment=assignment):
                results = self._run("per_user", enrollment=assignment)
                self.assertEqual(len(results), 2)
                self.assertTrue(all(r["enrollment"] == assignment for r in results))


if __name__ == "__main__":
    unittest.main()
