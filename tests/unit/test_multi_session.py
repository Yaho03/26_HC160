"""여러 세션을 합쳐 분석할 때의 경계 조건.

sample_id는 세션 안에서만 고유하다. 세션 경계를 넘는 집계도 성립하지 않는다.
두 문제 모두 단일 세션에서는 드러나지 않는다.
"""

import unittest

import numpy as np

from src.verification.defenses.probe_threshold import _aggregate, aggregate_by_session


class SampleKeyTest(unittest.TestCase):
    def test_sample_ids_collide_across_sessions(self):
        """계측 도구가 프레임 번호로 sample_id를 만들므로 세션마다 같은 값이 나온다."""
        from src.verification.defenses.probe_capture import sample_id

        self.assertEqual(sample_id(0, "clean"), sample_id(0, "clean"))

    def test_rows_are_keyed_by_session_and_sample(self):
        from src.verification.defenses.probe_threshold import sample_key

        left = {"session_id": "s1", "sample_id": "f000000_clean"}
        right = {"session_id": "s2", "sample_id": "f000000_clean"}

        self.assertNotEqual(sample_key(left), sample_key(right))


class SessionBoundaryTest(unittest.TestCase):
    def test_windows_do_not_span_two_sessions(self):
        """세션이 다르면 조명도 시점도 다르다. 한 윈도로 묶으면 의미가 없다."""
        values = [1.0, 1.0, 1.0, 9.0, 9.0, 9.0]
        sessions = ["s1", "s1", "s1", "s2", "s2", "s2"]

        combined = aggregate_by_session(values, sessions, window_frames=3)

        # 각 세션에서 1개씩, 경계를 넘는 윈도는 만들어지지 않는다
        self.assertEqual(len(combined), 2)
        self.assertEqual(sorted(combined.tolist()), [1.0, 9.0])

    def test_single_session_matches_plain_aggregation(self):
        values = [1.0, 5.0, 2.0, 8.0, 3.0]
        sessions = ["s1"] * 5

        self.assertTrue(
            np.allclose(
                aggregate_by_session(values, sessions, window_frames=3),
                _aggregate(np.asarray(values, float), 3),
            )
        )

    def test_session_shorter_than_the_window_is_dropped(self):
        """윈도를 채우지 못하는 세션은 집계 단위를 만들 수 없다."""
        values = [1.0, 2.0, 3.0, 4.0, 9.0]
        sessions = ["s1", "s1", "s1", "s1", "s2"]

        combined = aggregate_by_session(values, sessions, window_frames=3)
        self.assertEqual(len(combined), 2)
        self.assertNotIn(9.0, combined.tolist())

    def test_window_one_keeps_every_sample(self):
        values = [1.0, 2.0, 3.0]
        sessions = ["s1", "s2", "s3"]

        self.assertEqual(len(aggregate_by_session(values, sessions, window_frames=1)), 3)


if __name__ == "__main__":
    unittest.main()
