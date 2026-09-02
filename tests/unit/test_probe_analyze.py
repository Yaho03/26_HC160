import csv
import tempfile
import unittest
from pathlib import Path

from src.verification.defenses.probe_analyze import (
    ThresholdFromAttackDataError,
    combine_clean_normalized,
    detector_metrics,
    feature_table,
    load_probe_rows,
    roc_auc,
    threshold_at_fpr,
)
from src.verification.defenses.probe_log import PROBE_COLUMNS


class RocAucTest(unittest.TestCase):
    def test_perfect_separation(self):
        self.assertEqual(roc_auc([3.0, 4.0], [1.0, 2.0]), 1.0)

    def test_reversed_separation(self):
        self.assertEqual(roc_auc([1.0, 2.0], [3.0, 4.0]), 0.0)

    def test_complete_overlap_is_chance(self):
        self.assertEqual(roc_auc([1.0, 2.0], [1.0, 2.0]), 0.5)

    def test_undefined_when_a_class_is_empty(self):
        self.assertIsNone(roc_auc([], [1.0]))


class ThresholdTest(unittest.TestCase):
    def test_threshold_uses_clean_quantile_only(self):
        clean = [float(value) for value in range(100)]
        self.assertEqual(threshold_at_fpr(clean, 0.05), 95.0)

    def test_realized_fpr_never_exceeds_the_target(self):
        """보간된 임계값은 목표 FPR을 초과할 수 있으므로 관측값을 고른다."""
        import numpy as np

        rng = np.random.default_rng(0)
        for n in (37, 100, 251):
            clean = list(rng.normal(size=n))
            for target in (0.01, 0.05, 0.10):
                with self.subTest(n=n, target=target):
                    threshold = threshold_at_fpr(clean, target)
                    realized = sum(1 for value in clean if value >= threshold) / n
                    self.assertLessEqual(realized, target + 1e-12)

    def test_rejects_being_handed_labelled_attack_rows(self):
        """07 5절: 임계값은 normal session만으로 정한다."""
        with self.assertRaises(ThresholdFromAttackDataError):
            threshold_at_fpr([1.0, 2.0], 0.05, labels=["clean", "adversarial"])

    def test_undefined_for_empty_clean_set(self):
        self.assertIsNone(threshold_at_fpr([], 0.05))


class DetectorMetricsTest(unittest.TestCase):
    def test_counts_and_rates(self):
        result = detector_metrics(clean=[0.0, 0.0, 0.0, 9.0], adversarial=[9.0, 9.0, 0.0], threshold=5.0)

        self.assertEqual(result["true_positive"], 2)
        self.assertEqual(result["false_negative"], 1)
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["true_negative"], 3)
        self.assertAlmostEqual(result["tpr"], 2 / 3)
        self.assertAlmostEqual(result["fpr"], 1 / 4)
        self.assertAlmostEqual(result["precision"], 2 / 3)

    def test_zero_denominator_is_undefined_not_zero(self):
        """09 1절: 분모가 0이면 0이 아니라 undefined를 반환한다."""
        result = detector_metrics(clean=[], adversarial=[9.0], threshold=5.0)
        self.assertIsNone(result["fpr"])


class LoadProbeRowsTest(unittest.TestCase):
    def _write(self, path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=PROBE_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _row(self, **overrides):
        row = {
            "session_id": "s1", "subject_id": "p01", "sample_id": "f000_clean",
            "frame_idx": 0, "frame_ts_ms": 0.0, "dropped_frames": 0,
            "label": "clean", "transform": "blur0.8",
            "cos_orig_enroll": 0.8, "cos_transformed_enroll": 0.7,
            "cos_orig_transformed": 0.95, "embed_ms": 10.0,
        }
        row.update(overrides)
        return row

    def test_derives_both_measures_from_raw_cosines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.csv"
            self._write(path, [self._row()])
            rows = load_probe_rows(path)

        self.assertAlmostEqual(rows[0]["self_consistency"], 1.0 - 0.95)
        self.assertAlmostEqual(rows[0]["template_shift"], abs(0.8 - 0.7))

    def test_feature_table_splits_by_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.csv"
            self._write(path, [
                self._row(),
                self._row(sample_id="f000_adv", label="adversarial", cos_orig_transformed=0.5),
            ])
            table = feature_table(load_probe_rows(path))

        key = ("blur0.8", "self_consistency")
        self.assertEqual(len(table[key]["clean"]), 1)
        self.assertEqual(len(table[key]["adversarial"]), 1)


class CombineTest(unittest.TestCase):
    def test_normalizes_using_clean_statistics_only(self):
        table = {
            ("a", "self_consistency"): {"clean": [0.0, 2.0], "adversarial": [4.0]},
            ("b", "template_shift"): {"clean": [0.0, 2.0], "adversarial": [4.0]},
        }
        clean, adversarial = combine_clean_normalized(table, list(table))

        # clean 평균 1.0, 표준편차 1.0 → z = -1, +1 이 두 특징에서 더해진다
        self.assertAlmostEqual(clean[0], -2.0)
        self.assertAlmostEqual(clean[1], 2.0)
        self.assertAlmostEqual(adversarial[0], 6.0)


if __name__ == "__main__":
    unittest.main()
