import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.face_auth.evaluation.pad_cli import (
    _atomic_json_write,
    _parser,
    _prepare_registration,
    _validate_args,
    _validate_output_path,
    _verify_inputs_unchanged,
    main,
)
from src.common.reproducibility import sha256_file
from src.experiments.artifact_registration import registration_outputs


def required_arguments():
    return [
        "--manifest",
        "manifest.csv",
        "--manifest-id",
        "pad-manifest-v1",
        "--artifact-root",
        "dataset",
        "--pad-model",
        "pad.onnx",
        "--pad-model-version",
        "pad-v1",
        "--live-threshold",
        "0.8",
        "--threshold-version",
        "threshold-v1",
        "--run-id",
        "run-pad-v1",
        "--output",
        "report.json",
    ]


class PADCLIArgumentsTest(unittest.TestCase):
    def test_onnx_runtime_leaves_model_contract_defaults_to_factory(self):
        args = _parser().parse_args(
            required_arguments()
            + ["--pad-runtime", "onnx", "--pad-provider", "CPUExecutionProvider"]
        )
        _validate_args(args)
        self.assertIsNone(args.pad_input_size)
        self.assertIsNone(args.pad_live_class_index)
        self.assertIsNone(args.pad_output_kind)
        self.assertEqual(args.pad_provider, ["CPUExecutionProvider"])

    def test_negative_live_class_index_is_rejected(self):
        args = _parser().parse_args(
            required_arguments() + ["--pad-live-class-index", "-1"]
        )
        with self.assertRaisesRegex(SystemExit, "must not be negative"):
            _validate_args(args)

    def test_existing_report_is_refused_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            output.write_text("original", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                _validate_output_path(
                    output,
                    inputs=(Path(directory) / "manifest.csv",),
                    overwrite=False,
                )
            self.assertEqual(output.read_text(encoding="utf-8"), "original")

    def test_atomic_writer_emits_canonical_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "report.json"
            _atomic_json_write(output, {"b": 2, "a": 1}, overwrite=False)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")), {"a": 1, "b": 2}
            )
            self.assertEqual(output.read_text(encoding="utf-8"), '{"a":1,"b":2}\n')

    def test_changed_video_is_rejected_before_report_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            model = root / "pad.onnx"
            video = root / "sample.mp4"
            manifest.write_bytes(b"manifest-v1")
            model.write_bytes(b"model-v1")
            video.write_bytes(b"video-v1")
            original_video_hash = sha256_file(video)
            video.write_bytes(b"video-v2")
            with self.assertRaisesRegex(SystemExit, "source video changed"):
                _verify_inputs_unchanged(
                    manifest_path=manifest,
                    manifest_sha256=sha256_file(manifest),
                    model_path=model,
                    model_sha256=sha256_file(model),
                    artifact_root=root,
                    samples=[
                        {
                            "relative_video_path": "sample.mp4",
                            "video_sha256": original_video_hash,
                            "video_bytes": len(b"video-v1"),
                        }
                    ],
                )

    def test_registration_context_run_id_must_match_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            context = Path(directory) / "registration.json"
            context.write_text(
                json.dumps(registration_context(run_id="another-run")),
                encoding="utf-8",
            )
            args = _parser().parse_args(
                required_arguments() + ["--registration-context", str(context)]
            )
            with self.assertRaisesRegex(SystemExit, "run_id must match"):
                _prepare_registration(args, Path(directory) / "report.json")

    def test_main_registers_the_emitted_pad_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            model = root / "pad.onnx"
            output = root / "report.json"
            context = root / "registration.json"
            manifest.write_bytes(b"manifest-v1")
            model.write_bytes(b"model-v1")
            context.write_text(
                json.dumps(registration_context()), encoding="utf-8"
            )
            arguments = [
                "pad_cli",
                "--manifest",
                str(manifest),
                "--manifest-id",
                "pad-manifest-v1",
                "--artifact-root",
                str(root),
                "--pad-model",
                str(model),
                "--pad-model-version",
                "pad-v1",
                "--live-threshold",
                "0.8",
                "--threshold-version",
                "threshold-v1",
                "--run-id",
                "run-pad-v1",
                "--output",
                str(output),
                "--registration-context",
                str(context),
            ]
            scorer = Mock()
            scorer.metadata.return_value = {
                "runtime": "torchscript",
                "version": "pad-v1",
            }
            result = SimpleNamespace(to_dict=lambda: {"sample_id": "sample-1"})
            metrics = {"sample_counts": {"evaluated": 1}}
            with (
                patch("sys.argv", arguments),
                patch(
                    "src.face_auth.evaluation.pad_cli.git_state",
                    return_value={"git_commit": "b" * 40, "dirty_worktree": False},
                ),
                patch(
                    "src.face_auth.evaluation.pad_cli.load_pad_manifest",
                    return_value=[object()],
                ),
                patch("src.face_auth.evaluation.pad_cli.validate_pad_manifest"),
                patch(
                    "src.face_auth.evaluation.pad_cli.create_pad_scorer",
                    return_value=scorer,
                ),
                patch("src.face_auth.evaluation.pad_cli.MTCNNFaceDetector"),
                patch("src.face_auth.evaluation.pad_cli.PADVideoEvaluator"),
                patch(
                    "src.face_auth.evaluation.pad_cli.evaluate_pad_manifest",
                    return_value=[result],
                ),
                patch(
                    "src.face_auth.evaluation.pad_cli._verify_inputs_unchanged"
                ),
                patch(
                    "src.face_auth.evaluation.pad_cli.pad_metrics",
                    return_value=metrics,
                ),
                redirect_stdout(io.StringIO()),
            ):
                exit_code = main()

            outputs = registration_outputs(output)
            reference = json.loads(outputs.artifact_reference.read_text())
            run_manifest = json.loads(outputs.run_manifest.read_text())
            self.assertEqual(exit_code, 0)
            self.assertEqual(reference["kind"], "report")
            self.assertEqual(reference["sha256"], sha256_file(output))
            self.assertEqual(run_manifest["run_id"], "run-pad-v1")
            self.assertEqual(run_manifest["output_artifact_ids"], ["pad-report-001"])


def registration_context(**changes):
    value = {
        "schema_version": "1.0",
        "run_id": "run-pad-v1",
        "experiment_id": "EXP-PAD-001",
        "requirement_ids": ["FR-201"],
        "environment_sha256": "a" * 64,
        "seed": 42,
        "input_artifact_ids": ["pad-manifest-v1", "pad-model-v1"],
        "reproduce_command": "python -m src.face_auth.evaluation.pad_cli ...",
        "artifact_id": "pad-report-001",
        "relative_uri": "reports/pad-report-001.json",
    }
    value.update(changes)
    return value


if __name__ == "__main__":
    unittest.main()
