import json
import tempfile
import unittest
from pathlib import Path

from src.face_auth.evaluation.pad_cli import (
    _atomic_json_write,
    _parser,
    _validate_args,
    _validate_output_path,
    _verify_inputs_unchanged,
)
from src.common.reproducibility import sha256_file


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


if __name__ == "__main__":
    unittest.main()
