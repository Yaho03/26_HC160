import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from src.common.reproducibility import sha256_file
from src.evaluation.verification_calibration import (
    CalibrationError,
    UnsupportedFarTarget,
    VerificationScore,
    calibrate_threshold,
    clean_baseline_report_dict,
    evaluate_clean_baseline,
    far_target_supported,
    roc_auc,
    threshold_artifact_dict,
)
from src.evaluation.verification_baseline_cli import (
    _load_score_export_metadata,
    _load_scores,
    main,
)


class VerificationCalibrationTest(unittest.TestCase):
    def test_target_far_selects_best_tar_with_supported_far(self):
        records = _records(
            "calibration",
            genuine=(0.9, 0.8),
            impostor=(0.7, 0.4, 0.3, 0.2),
        )
        result = calibrate_threshold(records, method="target_far", target_far=0.25)
        self.assertEqual(result.threshold, 0.7)
        self.assertEqual(result.rates.far.value, 0.25)
        self.assertEqual(result.rates.tar.value, 1.0)

    def test_unsupported_far_target_is_rejected(self):
        records = _records(
            "calibration",
            genuine=(0.9,),
            impostor=(0.4, 0.3, 0.2, 0.1),
        )
        self.assertFalse(far_target_supported(0.1, 4))
        with self.assertRaises(UnsupportedFarTarget):
            calibrate_threshold(records, method="target_far", target_far=0.1)

    def test_eer_calibration_is_deterministic(self):
        records = _records(
            "calibration",
            genuine=(0.9, 0.6),
            impostor=(0.7, 0.2),
        )
        result = calibrate_threshold(records, method="eer")
        self.assertEqual(result.threshold, 0.7)
        self.assertEqual(result.eer.value, 0.5)

    def test_roc_auc_handles_tied_scores(self):
        records = _records(
            "calibration",
            genuine=(0.8, 0.5),
            impostor=(0.5, 0.2),
        )
        self.assertEqual(roc_auc(records), 0.875)

    def test_test_evaluation_uses_frozen_threshold_and_disjoint_pairs(self):
        calibration_records = _records(
            "calibration",
            genuine=(0.9, 0.8),
            impostor=(0.7, 0.4, 0.3, 0.2),
        )
        calibration = calibrate_threshold(
            calibration_records,
            method="target_far",
            target_far=0.25,
        )
        test_records = _records(
            "test",
            genuine=(0.85, 0.65),
            impostor=(0.75, 0.1),
            prefix="test",
        )
        result = evaluate_clean_baseline(
            calibration,
            test_records,
            calibration_pair_ids=(record.pair_id for record in calibration_records),
        )
        self.assertEqual(result.threshold, calibration.threshold)
        self.assertEqual(result.rates.counts.true_positive, 1)
        self.assertEqual(result.rates.counts.false_negative, 1)
        self.assertEqual(result.rates.counts.false_positive, 1)
        self.assertEqual(result.rates.counts.true_negative, 1)
        self.assertLessEqual(
            result.confidence_intervals.far.lower,
            result.rates.far.value,
        )
        self.assertGreaterEqual(
            result.confidence_intervals.far.upper,
            result.rates.far.value,
        )

    def test_calibration_test_overlap_is_rejected(self):
        calibration_records = _records(
            "calibration",
            genuine=(0.9,),
            impostor=(0.1,),
        )
        calibration = calibrate_threshold(calibration_records, method="eer")
        test_records = list(
            _records("test", genuine=(0.8,), impostor=(0.2,), prefix="test")
        )
        test_records[0] = VerificationScore(
            **{
                **test_records[0].__dict__,
                "pair_id": calibration_records[0].pair_id,
            }
        )
        with self.assertRaisesRegex(CalibrationError, "overlap"):
            evaluate_clean_baseline(
                calibration,
                test_records,
                calibration_pair_ids=(record.pair_id for record in calibration_records),
            )

    def test_protocol_mismatch_is_rejected(self):
        calibration_records = _records(
            "calibration",
            genuine=(0.9,),
            impostor=(0.1,),
        )
        calibration = calibrate_threshold(calibration_records, method="eer")
        test_records = tuple(
            VerificationScore(
                **{**record.__dict__, "model_artifact_id": "different-model"}
            )
            for record in _records("test", genuine=(0.8,), impostor=(0.2,), prefix="test")
        )
        with self.assertRaisesRegex(CalibrationError, "differs"):
            evaluate_clean_baseline(
                calibration,
                test_records,
                calibration_pair_ids=(record.pair_id for record in calibration_records),
            )

    def test_non_finite_scores_are_rejected(self):
        records = list(_records("calibration", genuine=(0.9,), impostor=(0.1,)))
        records[0] = VerificationScore(**{**records[0].__dict__, "score": math.nan})
        with self.assertRaisesRegex(CalibrationError, "finite"):
            calibrate_threshold(records, method="eer")

    def test_out_of_range_cosine_score_is_rejected(self):
        records = list(_records("calibration", genuine=(0.9,), impostor=(0.1,)))
        records[0] = VerificationScore(**{**records[0].__dict__, "score": 1.1})
        with self.assertRaisesRegex(CalibrationError, "between -1 and 1"):
            calibrate_threshold(records, method="eer")

    def test_threshold_artifact_binds_protocol_and_provenance(self):
        records = _records(
            "calibration",
            genuine=(0.9, 0.8),
            impostor=(0.7, 0.4, 0.3, 0.2),
        )
        calibration = calibrate_threshold(records, method="target_far", target_far=0.25)
        artifact = threshold_artifact_dict(
            calibration,
            threshold_artifact_id="thr_facenet_example",
            calibration_manifest_id="pairs_calibration_v1",
            calibration_manifest_sha256="a" * 64,
            model_artifact_sha256="c" * 64,
            preprocessing_artifact_sha256="d" * 64,
            calibration_score_export_sha256="e" * 64,
            created_by_run_id="run_exp_ver_001",
            created_at="2026-08-22T00:00:00Z",
        )
        self.assertEqual(artifact["threshold"], calibration.threshold)
        self.assertEqual(artifact["preprocessing_artifact_id"], "preprocess-v1")
        self.assertEqual(artifact["achieved_far"]["denominator"], 4)

    def test_clean_report_keeps_threshold_and_intervals(self):
        calibration_records = _records(
            "calibration",
            genuine=(0.9, 0.8),
            impostor=(0.7, 0.4, 0.3, 0.2),
        )
        calibration = calibrate_threshold(
            calibration_records,
            method="target_far",
            target_far=0.25,
        )
        test_records = _records(
            "test",
            genuine=(0.85, 0.65),
            impostor=(0.75, 0.1),
            prefix="test",
        )
        result = evaluate_clean_baseline(
            calibration,
            test_records,
            calibration_pair_ids=(record.pair_id for record in calibration_records),
        )
        report = clean_baseline_report_dict(
            result,
            calibration,
            threshold_artifact_id="thr-v1",
            test_manifest_id="test-pairs-v1",
            test_manifest_sha256="b" * 64,
            model_artifact_sha256="c" * 64,
            preprocessing_artifact_sha256="d" * 64,
            test_score_export_sha256="e" * 64,
            generated_by_run_id="run-v1",
            created_at="2026-08-22T00:00:00Z",
        )
        self.assertEqual(report["frozen_threshold"], calibration.threshold)
        self.assertIn("far", report["wilson_95_intervals"])
        self.assertEqual(report["rates"]["far"]["denominator"], 2)

    def test_cli_writes_threshold_and_clean_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = root / "calibration.jsonl"
            test_path = root / "test.jsonl"
            threshold_path = root / "threshold.json"
            report_path = root / "report.json"
            calibration_metadata_path = root / "calibration-metadata.json"
            test_metadata_path = root / "test-metadata.json"
            _write_scores(
                calibration_path,
                _records(
                    "calibration",
                    genuine=(0.9, 0.8),
                    impostor=(0.7, 0.4, 0.3, 0.2),
                ),
            )
            _write_scores(
                test_path,
                _records(
                    "test",
                    genuine=(0.85, 0.65),
                    impostor=(0.75, 0.1),
                    prefix="test",
                ),
            )
            _write_score_metadata(
                calibration_metadata_path,
                score_path=calibration_path,
                split="calibration",
                pair_manifest_id="cal-v1",
                pair_manifest_sha256="a" * 64,
            )
            _write_score_metadata(
                test_metadata_path,
                score_path=test_path,
                split="test",
                pair_manifest_id="test-v1",
                pair_manifest_sha256="b" * 64,
            )
            printed = io.StringIO()
            with redirect_stdout(printed):
                result = main(
                    [
                        "--calibration-scores", str(calibration_path),
                        "--calibration-score-metadata", str(calibration_metadata_path),
                        "--test-scores", str(test_path),
                        "--test-score-metadata", str(test_metadata_path),
                        "--selection-method", "target_far",
                        "--target-far", "0.25",
                        "--threshold-artifact-id", "thr-v1",
                        "--run-id", "run-v1",
                        "--threshold-output", str(threshold_path),
                        "--report-output", str(report_path),
                        "--created-at", "2026-08-22T00:00:00Z",
                    ]
                )
            self.assertEqual(result, 0)
            self.assertEqual(json.loads(printed.getvalue())["threshold"], 0.7)
            threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(threshold["threshold_artifact_id"], "thr-v1")
            self.assertEqual(report["threshold_artifact_id"], "thr-v1")
            self.assertEqual(report["test_pair_count"], 4)

            schema_root = Path(__file__).parents[2] / "schemas"
            threshold_schema = json.loads(
                (schema_root / "threshold-artifact.schema.json").read_text(encoding="utf-8")
            )
            report_schema = json.loads(
                (schema_root / "clean-verification-report.schema.json").read_text(
                    encoding="utf-8"
                )
            )
            score_schema = json.loads(
                (schema_root / "verification-score.schema.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(threshold), set(threshold_schema["required"]))
            self.assertEqual(set(report), set(report_schema["required"]))
            self.assertEqual(
                set(VerificationScore.__dataclass_fields__),
                set(score_schema["required"]),
            )

    def test_score_tampering_breaks_export_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scores = root / "scores.jsonl"
            metadata = root / "metadata.json"
            _write_scores(
                scores,
                _records("calibration", genuine=(0.9,), impostor=(0.1,)),
            )
            _write_score_metadata(
                metadata,
                score_path=scores,
                split="calibration",
                pair_manifest_id="cal-v1",
                pair_manifest_sha256="a" * 64,
            )
            scores.write_text(
                scores.read_text(encoding="utf-8").replace("0.9", "0.8"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CalibrationError, "hash differs"):
                _load_score_export_metadata(
                    metadata,
                    loaded_score_file=_load_scores(scores),
                    expected_split="calibration",
                )


def _records(
    split: str,
    *,
    genuine: tuple[float, ...],
    impostor: tuple[float, ...],
    prefix: str = "cal",
) -> tuple[VerificationScore, ...]:
    values = [(True, score) for score in genuine]
    values.extend((False, score) for score in impostor)
    return tuple(
        VerificationScore(
            pair_id=f"{prefix}-{index}",
            same_identity=same_identity,
            score=score,
            split=split,
            protocol_id="facenet-vggface2-v1",
            model_artifact_id="model-v1",
            preprocessing_artifact_id="preprocess-v1",
        )
        for index, (same_identity, score) in enumerate(values)
    )


def _write_scores(path: Path, records: tuple[VerificationScore, ...]) -> None:
    path.write_text(
        "".join(json.dumps(record.__dict__, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_score_metadata(
    path: Path,
    *,
    score_path: Path,
    split: str,
    pair_manifest_id: str,
    pair_manifest_sha256: str,
) -> None:
    row_count = sum(
        1 for line in score_path.read_text(encoding="utf-8").splitlines() if line
    )
    value = {
        "schema_version": "1.0",
        "export_id": f"export-{split}",
        "run_id": f"run-{split}",
        "created_at": "2026-08-22T00:00:00Z",
        "code": {"git_commit": "f" * 40, "dirty_worktree": False},
        "protocol_id": "facenet-vggface2-v1",
        "split": split,
        "software": {
            "python": "3.9.0",
            "torch": "2.2.2",
            "facenet_pytorch": "2.6.0",
            "numpy": "1.26.4",
            "pillow": "10.2.0",
        },
        "model_artifact": {
            "id": "model-v1",
            "sha256": "c" * 64,
            "bytes": 1,
            "architecture": "facenet_pytorch.InceptionResnetV1",
            "weights_source": "vggface2",
        },
        "preprocessing_artifact": {
            "id": "preprocess-v1",
            "sha256": "d" * 64,
            "bytes": 1,
        },
        "dataset_manifest": {"id": "dataset-v1", "sha256": "e" * 64, "row_count": 4},
        "pair_manifest": {
            "id": pair_manifest_id,
            "sha256": pair_manifest_sha256,
            "row_count": row_count,
            "split": split,
        },
        "score_file": {
            "sha256": sha256_file(score_path),
            "bytes": score_path.stat().st_size,
            "row_count": row_count,
        },
        "evaluation_config": {
            "device": "cpu",
            "batch_size": 4,
            "require_identity_disjoint": True,
            "deterministic_algorithms": True,
        },
    }
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
