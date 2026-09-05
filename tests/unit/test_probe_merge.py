"""여러 세션의 계측 CSV 합치기.

컬럼이 세션마다 다를 수 있다. attack_kind는 나중에 추가된 컬럼이므로 그 이전
세션에는 없다. 소급 적용하지 않되 합칠 때는 정규화한다.
"""

import csv
import tempfile
import unittest
from pathlib import Path

from src.verification.defenses.probe_log import PROBE_COLUMNS
from src.verification.defenses.probe_merge import DuplicateSessionError, merge_probe_csvs


def _write(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _row(session, subject, label, **extra):
    row = {
        "session_id": session, "subject_id": subject, "sample_id": "f000_x",
        "frame_idx": 0, "frame_ts_ms": 0.0, "dropped_frames": 0,
        "label": label, "transform": "blur0.8",
        "cos_orig_enroll": 0.8, "cos_transformed_enroll": 0.7,
        "cos_orig_transformed": 0.95, "embed_ms": 10.0,
    }
    row.update(extra)
    return row


class MergeTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)

    def tearDown(self):
        self._dir.cleanup()

    def test_older_csv_without_attack_kind_is_normalised(self):
        old_columns = [c for c in PROBE_COLUMNS if c != "attack_kind"]
        old = self.root / "old.csv"
        _write(old, [_row("s1", "p01", "adversarial")], old_columns)

        new = self.root / "new.csv"
        _write(new, [_row("s2", "p01", "adversarial", attack_kind="fgsm")], PROBE_COLUMNS)

        out = merge_probe_csvs([old, new], self.root / "merged.csv")
        rows = list(csv.DictReader(out.open(encoding="utf-8")))

        self.assertEqual(tuple(rows[0]), PROBE_COLUMNS)
        kinds = {row["session_id"]: row["attack_kind"] for row in rows}
        self.assertEqual(kinds["s1"], "unspecified")
        self.assertEqual(kinds["s2"], "fgsm")

    def test_clean_rows_keep_an_empty_attack_kind(self):
        old_columns = [c for c in PROBE_COLUMNS if c != "attack_kind"]
        old = self.root / "old.csv"
        _write(old, [_row("s1", "p01", "clean")], old_columns)

        out = merge_probe_csvs([old], self.root / "merged.csv")
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        self.assertEqual(rows[0]["attack_kind"], "")

    def test_duplicate_session_is_rejected(self):
        """같은 세션을 두 번 넣으면 표본이 부풀려진다."""
        first = self.root / "a.csv"
        second = self.root / "b.csv"
        _write(first, [_row("s1", "p01", "clean")], PROBE_COLUMNS)
        _write(second, [_row("s1", "p01", "clean")], PROBE_COLUMNS)

        with self.assertRaises(DuplicateSessionError):
            merge_probe_csvs([first, second], self.root / "merged.csv")

    def test_reports_sessions_and_subjects(self):
        first = self.root / "a.csv"
        second = self.root / "b.csv"
        _write(first, [_row("s1", "p01", "clean")], PROBE_COLUMNS)
        _write(second, [_row("s2", "p03", "clean")], PROBE_COLUMNS)

        out = merge_probe_csvs([first, second], self.root / "merged.csv")
        rows = list(csv.DictReader(out.open(encoding="utf-8")))

        self.assertEqual({r["session_id"] for r in rows}, {"s1", "s2"})
        self.assertEqual({r["subject_id"] for r in rows}, {"p01", "p03"})

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            merge_probe_csvs([], self.root / "merged.csv")


if __name__ == "__main__":
    unittest.main()
