import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.common.reproducibility import sha256_file, stable_json_bytes
from src.datasets.manifest import DatasetManifestRow
from src.evaluation.facenet_score_export_cli import (
    FaceNetVGGFace2Embedder,
    ScoreExportError,
    VerificationPair,
    load_preprocessing_config,
    main,
    score_verification_pairs,
    validate_pair_protocol,
)


class _FakeFaceNetEmbedder:
    def __init__(self, **kwargs):
        pass

    def embed_many(self, paths):
        vectors = {
            "a1.png": np.array([1.0, 0.0], dtype=np.float32),
            "a2.png": np.array([0.9, 0.1], dtype=np.float32),
            "b1.png": np.array([0.0, 1.0], dtype=np.float32),
        }
        return [vectors[path.name] for path in paths]


class FaceNetScoreExportTest(unittest.TestCase):
    @patch("torch.use_deterministic_algorithms")
    @patch("torch.load", return_value={"weights": "fixture"})
    @patch("facenet_pytorch.InceptionResnetV1")
    def test_checkpoint_loads_training_head_before_embedding_mode(
        self, model_class, load_checkpoint, deterministic
    ):
        model = model_class.return_value
        model.to.return_value = model
        FaceNetVGGFace2Embedder(
            checkpoint=Path("facenet.pt"),
            preprocessing={},
            device="cpu",
            batch_size=2,
        )
        model_class.assert_called_once_with(
            pretrained=None,
            classify=True,
            num_classes=8631,
        )
        load_checkpoint.assert_called_once_with(
            Path("facenet.pt"), map_location="cpu", weights_only=True
        )
        model.load_state_dict.assert_called_once_with(
            {"weights": "fixture"}, strict=True
        )
        deterministic.assert_called_once_with(True)
        self.assertIs(model.classify, False)

    def test_pair_labels_must_match_pseudonymous_dataset_identities(self):
        rows = _dataset_rows("calibration")
        pair = _pair(
            "pair_bad_label",
            "sample_a1",
            "sample_b1",
            same_identity=True,
        )
        with self.assertRaisesRegex(ScoreExportError, "contradicts"):
            validate_pair_protocol(
                (pair,), protocol_id="facenet-vggface2-v1", dataset_rows=rows
            )

    def test_score_export_caches_samples_and_emits_cosine_records(self):
        rows = _dataset_rows("calibration")
        pairs = (
            _pair("pair_genuine", "sample_a1", "sample_a2", same_identity=True),
            _pair("pair_impostor", "sample_a1", "sample_b1", same_identity=False),
        )
        calls = []

        def embed_many(paths):
            calls.append([path.name for path in paths])
            return _FakeFaceNetEmbedder().embed_many(paths)

        records = score_verification_pairs(
            pairs,
            rows,
            artifact_root=Path("dataset"),
            embed_many=embed_many,
            model_artifact_id="facenet-weights-v1",
            preprocessing_artifact_id="preprocess-v1",
        )

        self.assertEqual(calls, [["a1.png", "a2.png", "b1.png"]])
        self.assertGreater(records[0].score, 0.99)
        self.assertEqual(records[1].score, 0.0)
        self.assertEqual(records[0].split, "calibration")

    def test_preprocessing_config_is_frozen_to_supported_contract(self):
        config_path = (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "models"
            / "facenet-vggface2-preprocessing.json"
        )
        config = load_preprocessing_config(config_path)
        self.assertEqual(config["image_size"], [160, 160])
        with tempfile.TemporaryDirectory() as directory:
            changed = dict(config, pixel_divide=127.5)
            path = Path(directory) / "changed.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ScoreExportError, "unsupported"):
                load_preprocessing_config(path)

    @patch(
        "src.evaluation.facenet_score_export_cli.FaceNetVGGFace2Embedder",
        _FakeFaceNetEmbedder,
    )
    @patch(
        "src.evaluation.facenet_score_export_cli._read_code_state",
        return_value={"git_commit": "f" * 40, "dirty_worktree": False},
    )
    def test_cli_writes_scores_and_provenance_without_local_paths(self, _code_state):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_root = root / "dataset"
            rows = _write_dataset(artifact_root, split="calibration")
            dataset_manifest = root / "dataset.jsonl"
            dataset_manifest.write_bytes(
                b"".join(stable_json_bytes(row.to_dict()) + b"\n" for row in rows)
            )
            pairs = (
                _pair("pair_genuine", "sample_a1", "sample_a2", same_identity=True),
                _pair("pair_impostor", "sample_a1", "sample_b1", same_identity=False),
            )
            pair_manifest = root / "pairs.jsonl"
            pair_manifest.write_bytes(
                b"".join(stable_json_bytes(_pair_dict(pair)) + b"\n" for pair in pairs)
            )
            checkpoint = root / "facenet.pt"
            checkpoint.write_bytes(b"model-weights")
            preprocessing = (
                Path(__file__).resolve().parents[2]
                / "configs"
                / "models"
                / "facenet-vggface2-preprocessing.json"
            )
            scores_output = root / "scores.jsonl"
            metadata_output = root / "scores.metadata.json"

            printed = io.StringIO()
            with redirect_stdout(printed):
                result = main(
                    [
                    "--dataset-manifest",
                    str(dataset_manifest),
                    "--artifact-root",
                    str(artifact_root),
                    "--pair-manifest",
                    str(pair_manifest),
                    "--pair-manifest-id",
                    "pairs-calibration-v1",
                    "--protocol-id",
                    "facenet-vggface2-v1",
                    "--model-checkpoint",
                    str(checkpoint),
                    "--model-artifact-id",
                    "facenet-vggface2-weights-v1",
                    "--preprocessing-config",
                    str(preprocessing),
                    "--run-id",
                    "run-score-calibration-v1",
                    "--scores-output",
                    str(scores_output),
                    "--metadata-output",
                    str(metadata_output),
                    "--created-at",
                    "2026-08-22T00:00:00Z",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(json.loads(printed.getvalue())["pair_count"], 2)
            metadata = json.loads(metadata_output.read_text(encoding="utf-8"))
            self.assertEqual(metadata["score_file"]["sha256"], sha256_file(scores_output))
            self.assertEqual(metadata["pair_manifest"]["split"], "calibration")
            self.assertNotIn(str(root), metadata_output.read_text(encoding="utf-8"))
            schema = json.loads(
                (
                    Path(__file__).resolve().parents[2]
                    / "schemas"
                    / "verification-score-export.schema.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(set(metadata), set(schema["required"]))


def _dataset_rows(split: str) -> tuple[DatasetManifestRow, ...]:
    return (
        _row("sample_a1", "id_aaaaaaaaaaaaaaaa", f"{split}/id_aaaaaaaaaaaaaaaa/a1.png", split),
        _row("sample_a2", "id_aaaaaaaaaaaaaaaa", f"{split}/id_aaaaaaaaaaaaaaaa/a2.png", split),
        _row("sample_b1", "id_bbbbbbbbbbbbbbbb", f"{split}/id_bbbbbbbbbbbbbbbb/b1.png", split),
    )


def _row(sample_id: str, identity: str, relative_uri: str, split: str) -> DatasetManifestRow:
    return DatasetManifestRow(
        schema_version="1.0",
        dataset_id="dataset-v1",
        sample_id=sample_id,
        identity_token=identity,
        relative_uri=relative_uri,
        media_sha256="0" * 64,
        split=split,
        width_px=16,
        height_px=16,
        license_id="test-only",
    )


def _pair(
    pair_id: str,
    left: str,
    right: str,
    *,
    same_identity: bool,
    split: str = "calibration",
) -> VerificationPair:
    return VerificationPair(
        schema_version="1.0",
        pair_id=pair_id,
        protocol_id="facenet-vggface2-v1",
        left_sample_id=left,
        right_sample_id=right,
        same_identity=same_identity,
        split=split,
    )


def _pair_dict(pair: VerificationPair) -> dict:
    value = {
        "schema_version": pair.schema_version,
        "pair_id": pair.pair_id,
        "protocol_id": pair.protocol_id,
        "left_sample_id": pair.left_sample_id,
        "right_sample_id": pair.right_sample_id,
        "same_identity": pair.same_identity,
        "split": pair.split,
    }
    return value


def _write_dataset(root: Path, *, split: str) -> tuple[DatasetManifestRow, ...]:
    rows = []
    for row in _dataset_rows(split):
        path = root / row.relative_uri
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), (120, 100, 80)).save(path)
        rows.append(
            DatasetManifestRow(
                **{
                    **row.__dict__,
                    "media_sha256": sha256_file(path),
                }
            )
        )
    return tuple(rows)


if __name__ == "__main__":
    unittest.main()
