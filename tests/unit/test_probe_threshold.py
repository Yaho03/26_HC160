import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.verification.defenses.probe_log import PROBE_COLUMNS
from src.verification.defenses.probe_threshold import (
    InsufficientCleanSamplesError,
    build_artifact,
    derive_limitations,
)


def _write_probe(path, *, n_clean, n_adversarial, sessions=("s1",), subjects=("p01",)):
    rows = []
    index = 0
    for session in sessions:
        for subject in subjects:
            for i in range(n_clean):
                rows.append((session, subject, f"f{i:06d}_clean", i, "clean", 0.95 - i * 0.0001))
            for i in range(n_adversarial):
                rows.append((session, subject, f"f{i:06d}_adv", i, "adversarial", 0.5 - i * 0.001))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROBE_COLUMNS)
        writer.writeheader()
        for session, subject, sample_id, frame, label, cos_ot in rows:
            for transform in ("blur0.8", "jpeg_q75"):
                writer.writerow({
                    "session_id": session, "subject_id": subject, "sample_id": sample_id,
                    "frame_idx": frame, "frame_ts_ms": 0.0, "dropped_frames": 0,
                    "label": label, "transform": transform,
                    "cos_orig_enroll": 0.8, "cos_transformed_enroll": 0.8,
                    "cos_orig_transformed": cos_ot, "embed_ms": 10.0,
                })
            index += 1
    return Path(path)


def _sidecar(**overrides):
    meta = {
        "session_id": "s1", "subject_id": "p01",
        "model": {"name": "InceptionResnetV1", "pretrained": "vggface2",
                  "weights_file": "w.pt", "weights_sha256": "a" * 64,
                  "preprocess": "resize160"},
        "transforms": {"blur0.8": {"radius": 0.8}, "jpeg_q75": {"quality": 75}},
        "attack": {"kind": "pgd_targeted_enroll", "epsilon": 0.03, "steps": 40, "every": 5},
        "jpeg_headroom_q75": 2.2,
        "counters": {"samples_clean": 300, "samples_adversarial": 60},
    }
    meta.update(overrides)
    return meta


class BuildArtifactTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.probe = _write_probe(Path(self._dir.name) / "probe.csv", n_clean=300, n_adversarial=60)

    def tearDown(self):
        self._dir.cleanup()

    def test_calibration_and_evaluation_are_separated(self):
        artifact = build_artifact(self.probe, [_sidecar()], target_fpr=0.01, top_k=2)

        self.assertEqual(artifact["calibration"]["n_clean"], 300)
        self.assertEqual(artifact["evaluation"]["n_adversarial"], 60)
        self.assertNotIn("n_adversarial", artifact["calibration"])

    def test_achieved_fpr_never_exceeds_the_target(self):
        artifact = build_artifact(self.probe, [_sidecar()], target_fpr=0.01, top_k=2)
        self.assertLessEqual(artifact["calibration"]["achieved_fpr"], 0.01)

    def test_records_model_and_transform_provenance(self):
        artifact = build_artifact(self.probe, [_sidecar()], target_fpr=0.01, top_k=2)

        self.assertEqual(artifact["model"]["weights_sha256"], "a" * 64)
        self.assertIn("blur0.8", artifact["transforms"])
        self.assertRegex(artifact["calibration"]["probe_csv_sha256"], r"^[0-9a-f]{64}$")

    def test_refuses_when_clean_samples_cannot_reach_the_target_fpr(self):
        """표본이 부족하면 임계값을 만들지 않는다. 조용히 FPR 0으로 낮추지 않는다."""
        small = _write_probe(Path(self._dir.name) / "small.csv", n_clean=20, n_adversarial=5)
        with self.assertRaises(InsufficientCleanSamplesError):
            build_artifact(small, [_sidecar()], target_fpr=0.01, top_k=2)

    def test_selection_method_and_direction_are_fixed(self):
        artifact = build_artifact(self.probe, [_sidecar()], target_fpr=0.01, top_k=2)

        self.assertEqual(artifact["selection_method"], "target_fpr")
        self.assertEqual(artifact["score_direction"], "higher_is_adversarial")


class SchemaConformanceTest(unittest.TestCase):
    """생성기와 스키마가 어긋나면 여기서 잡힌다."""

    def test_generated_artifact_validates_against_the_schema(self):
        import jsonschema

        schema = json.loads(
            Path("schemas/detector-threshold-artifact.schema.json").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as directory:
            probe = _write_probe(Path(directory) / "probe.csv", n_clean=300, n_adversarial=60)
            artifact = build_artifact(probe, [_sidecar()], target_fpr=0.01, top_k=2)

        jsonschema.validate(artifact, schema)

    def test_limitations_survive_into_the_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            probe = _write_probe(Path(directory) / "probe.csv", n_clean=300, n_adversarial=60)
            artifact = build_artifact(probe, [_sidecar()], target_fpr=0.01, top_k=2)

        self.assertGreaterEqual(len(artifact["limitations"]), 3)


class LimitationsTest(unittest.TestCase):
    def test_single_subject_and_session_are_reported(self):
        limitations = derive_limitations([_sidecar()], subjects={"p01"}, sessions={"s1"})
        joined = " ".join(limitations)

        self.assertIn("피험자", joined)
        self.assertIn("세션", joined)

    def test_single_attack_kind_is_reported(self):
        limitations = derive_limitations([_sidecar()], subjects={"p01"}, sessions={"s1"})
        self.assertTrue(any("공격" in item for item in limitations))

    def test_adaptive_attack_is_always_reported_as_unevaluated(self):
        limitations = derive_limitations(
            [_sidecar(), _sidecar(session_id="s2", subject_id="p02")],
            subjects={"p01", "p02"},
            sessions={"s1", "s2"},
        )
        self.assertTrue(any("adaptive" in item.lower() for item in limitations))


if __name__ == "__main__":
    unittest.main()
