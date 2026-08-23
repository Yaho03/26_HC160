import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common.reproducibility import sha256_file
from src.experiments.artifact_registration import (
    ArtifactRegistrationError,
    RegistrationContext,
    load_registration_context,
    register_completed_output,
    registration_outputs,
)


SHA = "a" * 64
COMMIT = "b" * 40


def context(**changes):
    value = {
        "schema_version": "1.0",
        "run_id": "run-001",
        "experiment_id": "EXP-PAD-001",
        "requirement_ids": ["FR-201"],
        "environment_sha256": SHA,
        "seed": 7,
        "input_artifact_ids": ["manifest-001", "model-001"],
        "reproduce_command": "python -m evaluator --fixed-args",
        "artifact_id": "report-001",
        "relative_uri": "reports/report-001.json",
    }
    value.update(changes)
    return RegistrationContext.from_dict(value)


class RegistrationContextTest(unittest.TestCase):
    def test_context_loads_and_rejects_unknown_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.json"
            path.write_text(
                json.dumps(
                    {
                        **context().__dict__,
                        "requirement_ids": ["FR-201"],
                        "input_artifact_ids": ["manifest-001", "model-001"],
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_registration_context(path)
            self.assertEqual(loaded.artifact_id, "report-001")
            value = json.loads(path.read_text(encoding="utf-8"))
            value["local_username"] = "must-not-leak"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ArtifactRegistrationError, "unknown"):
                load_registration_context(path)

    def test_context_rejects_output_reused_as_input(self):
        with self.assertRaisesRegex(ArtifactRegistrationError, "must differ"):
            context(input_artifact_ids=["report-001"])


class ArtifactRegistrationTest(unittest.TestCase):
    def test_registers_reference_and_completed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "report.json"
            artifact.write_text('{"status":"ok"}\n', encoding="utf-8")
            outputs = register_completed_output(
                artifact,
                context=context(),
                kind="report",
                created_at="2026-08-23T10:00:00Z",
                config_sha256=SHA,
                git_commit=COMMIT,
                dirty_worktree=False,
                device={"type": "cpu"},
                started_at="2026-08-23T09:59:00Z",
                ended_at="2026-08-23T10:00:00Z",
            )
            reference = json.loads(outputs.artifact_reference.read_text())
            manifest = json.loads(outputs.run_manifest.read_text())
            self.assertEqual(reference["sha256"], sha256_file(artifact))
            self.assertEqual(reference["bytes"], artifact.stat().st_size)
            self.assertEqual(reference["producer_run_id"], "run-001")
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["output_artifact_ids"], ["report-001"])
            self.assertEqual(
                manifest["input_artifact_ids"], ["manifest-001", "model-001"]
            )

    def test_existing_sidecar_prevents_any_write(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "report.json"
            artifact.write_text("{}\n", encoding="utf-8")
            outputs = registration_outputs(artifact)
            outputs.run_manifest.write_text("original", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactRegistrationError, "overwrite"):
                register_completed_output(
                    artifact,
                    context=context(),
                    kind="report",
                    created_at="2026-08-23T10:00:00Z",
                    config_sha256=SHA,
                    git_commit=COMMIT,
                    dirty_worktree=False,
                    device={"type": "cpu"},
                    started_at="2026-08-23T09:59:00Z",
                    ended_at="2026-08-23T10:00:00Z",
                )
            self.assertFalse(outputs.artifact_reference.exists())
            self.assertEqual(outputs.run_manifest.read_text(), "original")

    def test_second_sidecar_failure_rolls_back_first(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "report.json"
            artifact.write_text("{}\n", encoding="utf-8")
            outputs = registration_outputs(artifact)
            from src.experiments import artifact_registration

            real_write = artifact_registration._write_new_json

            def fail_second(path, value):
                if path == outputs.run_manifest:
                    raise ArtifactRegistrationError("simulated failure")
                real_write(path, value)

            with (
                patch.object(
                    artifact_registration, "_write_new_json", side_effect=fail_second
                ),
                self.assertRaisesRegex(ArtifactRegistrationError, "simulated"),
            ):
                register_completed_output(
                    artifact,
                    context=context(),
                    kind="report",
                    created_at="2026-08-23T10:00:00Z",
                    config_sha256=SHA,
                    git_commit=COMMIT,
                    dirty_worktree=False,
                    device={"type": "cpu"},
                    started_at="2026-08-23T09:59:00Z",
                    ended_at="2026-08-23T10:00:00Z",
                )
            self.assertFalse(outputs.artifact_reference.exists())
            self.assertFalse(outputs.run_manifest.exists())


if __name__ == "__main__":
    unittest.main()
