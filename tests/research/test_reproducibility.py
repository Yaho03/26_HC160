import tempfile
import unittest
from pathlib import Path

from src.common.reproducibility import sha256_file, stable_json_bytes, stable_json_sha256
from src.experiments.run_manifest import RunManifest


class ReproducibilityTest(unittest.TestCase):
    def test_stable_json_hash_is_key_order_independent(self):
        left = {"b": 2, "a": [1, 3]}
        right = {"a": [1, 3], "b": 2}
        self.assertEqual(stable_json_bytes(left), stable_json_bytes(right))
        self.assertEqual(stable_json_sha256(left), stable_json_sha256(right))

    def test_file_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.bin"
            path.write_bytes(b"hc160")
            self.assertEqual(
                sha256_file(path),
                "e90f29be797a871bcb4a0e0c6aca8c3aa6fe552d896b1463f03f9a387df5e6ca",
            )

    def test_completed_run_requires_end_time(self):
        common = dict(
            run_id="run-1",
            experiment_id="EXP-VER-001",
            requirement_ids=("VER-001",),
            status="completed",
            config_sha256="a" * 64,
            git_commit="b" * 40,
            environment_sha256="c" * 64,
            seed=42,
            device={"type": "cpu"},
            started_at="2026-08-22T00:00:00Z",
            input_artifact_ids=(),
            output_artifact_ids=(),
            reproduce_command="python -m example",
        )
        with self.assertRaises(ValueError):
            RunManifest(**common)
        manifest = RunManifest(**common, ended_at="2026-08-22T00:01:00Z")
        self.assertEqual(manifest.to_dict()["requirement_ids"], ["VER-001"])


if __name__ == "__main__":
    unittest.main()
