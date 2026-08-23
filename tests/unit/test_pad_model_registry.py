import json
import tempfile
import unittest
from pathlib import Path

from src.common.reproducibility import sha256_file
from src.face_auth.inference.pad_model_registry import (
    load_pad_model_artifact,
    verify_pad_model_artifact,
)


def registry_payload(artifact: bytes) -> dict:
    import hashlib

    return {
        "schema_version": "1.0",
        "model_id": "pad-model-test-v1",
        "task": "presentation_attack_detection",
        "runtime": "onnx",
        "validation_status": "candidate_unvalidated",
        "artifact": {
            "filename": "pad.onnx",
            "source_url": "https://example.test/pad.onnx",
            "source_page": "https://example.test/model",
            "sha256": hashlib.sha256(artifact).hexdigest(),
            "bytes": len(artifact),
            "license": "MIT",
            "license_url": "https://example.test/license",
        },
        "input": {
            "name": "actual_input_1",
            "layout": "NCHW",
            "shape": [1, 3, 128, 128],
            "color_order": "RGB",
            "mean": [151.2405, 119.595, 107.8395],
            "scale": [63.0105, 56.457, 55.0035],
        },
        "output": {
            "name": "output1",
            "kind": "probability",
            "shape": [1, 2],
            "live_class_index": 0,
            "classes": {"0": "bona_fide", "1": "spoof"},
        },
    }


class PADModelRegistryTest(unittest.TestCase):
    def test_registered_artifact_hash_and_size_are_verified(self):
        artifact = b"fake-onnx-model"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry.json"
            model_path = root / "pad.onnx"
            registry.write_text(
                json.dumps(registry_payload(artifact)), encoding="utf-8"
            )
            model_path.write_bytes(artifact)

            model = load_pad_model_artifact(registry)
            result = verify_pad_model_artifact(model, model_path)

            self.assertTrue(result.verified)
            self.assertEqual(result.sha256, sha256_file(model_path))

    def test_changed_artifact_is_rejected_before_inference(self):
        artifact = b"fake-onnx-model"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry.json"
            model_path = root / "pad.onnx"
            registry.write_text(
                json.dumps(registry_payload(artifact)), encoding="utf-8"
            )
            model_path.write_bytes(b"tampered-model")

            model = load_pad_model_artifact(registry)
            with self.assertRaisesRegex(ValueError, "byte count mismatch"):
                verify_pad_model_artifact(model, model_path)

    def test_unknown_registry_field_is_rejected(self):
        payload = registry_payload(b"model")
        payload["typo_field"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"unknown=\['typo_field'\]"):
                load_pad_model_artifact(path)

    def test_live_class_must_map_to_bona_fide(self):
        payload = registry_payload(b"model")
        payload["output"]["live_class_index"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bona_fide"):
                load_pad_model_artifact(path)


if __name__ == "__main__":
    unittest.main()
