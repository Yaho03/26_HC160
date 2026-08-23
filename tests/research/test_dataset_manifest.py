import io
import json
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from src.datasets.manifest import (
    DatasetManifestError,
    build_snapshot_metadata,
    discover_manifest_rows,
    load_manifest,
    validate_manifest_rows,
    write_manifest,
)
from src.datasets.manifest_cli import main


class DatasetManifestTest(unittest.TestCase):
    def test_discovery_validation_and_round_trip_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            _write_png(root / "train" / "id_0123456789abcdef" / "a.png", 16, 12, b"a")
            _write_png(root / "test" / "id_fedcba9876543210" / "b.png", 20, 10, b"b")

            rows = discover_manifest_rows(root, "lfw-contract-v1", "LFW-terms")
            report = validate_manifest_rows(rows, artifact_root=root)
            self.assertEqual(report.row_count, 2)
            self.assertEqual(report.split_counts["train"], 1)
            self.assertEqual(report.split_counts["test"], 1)
            self.assertTrue(all(not Path(row.relative_uri).is_absolute() for row in rows))

            manifest = Path(directory) / "manifest.jsonl"
            written_hash = write_manifest(rows, manifest)
            self.assertEqual(written_hash, report.manifest_sha256)
            self.assertEqual(load_manifest(manifest), rows)

    def test_raw_identity_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_png(root / "train" / "Person Name" / "a.png", 8, 8, b"a")
            with self.assertRaisesRegex(DatasetManifestError, "pseudonymous"):
                discover_manifest_rows(root, "dataset", "license")

    def test_jpeg_dimensions_are_recorded_without_image_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_jpeg(
                root / "test" / "id_0123456789abcdef" / "a.jpg",
                width=31,
                height=17,
            )
            rows = discover_manifest_rows(root, "dataset", "license")
            self.assertEqual((rows[0].width_px, rows[0].height_px), (31, 17))

    def test_symlink_outside_artifact_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "artifacts"
            outside = base / "outside.png"
            _write_png(outside, 8, 8, b"outside")
            link = root / "test" / "id_0123456789abcdef" / "linked.png"
            link.parent.mkdir(parents=True)
            link.symlink_to(outside)
            with self.assertRaisesRegex(DatasetManifestError, "escapes root"):
                discover_manifest_rows(root, "dataset", "license")

    def test_nested_path_that_could_leak_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_png(
                root / "train" / "id_0123456789abcdef" / "Raw Person" / "a.png",
                8,
                8,
                b"a",
            )
            with self.assertRaisesRegex(DatasetManifestError, "exactly"):
                discover_manifest_rows(root, "dataset", "license")

    def test_same_media_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"same"
            _write_png(root / "train" / "id_0123456789abcdef" / "a.png", 8, 8, payload)
            _write_png(root / "test" / "id_fedcba9876543210" / "b.png", 8, 8, payload)
            rows = discover_manifest_rows(root, "dataset", "license")
            with self.assertRaisesRegex(DatasetManifestError, "duplicate sample_id|crosses splits"):
                validate_manifest_rows(rows)

    def test_identity_disjoint_policy_is_optional_and_enforceable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token = "id_0123456789abcdef"
            _write_png(root / "train" / token / "a.png", 8, 8, b"a")
            _write_png(root / "test" / token / "b.png", 8, 8, b"b")
            rows = discover_manifest_rows(root, "dataset", "license")
            validate_manifest_rows(rows, require_identity_disjoint=False)
            with self.assertRaisesRegex(DatasetManifestError, "identity crosses splits"):
                validate_manifest_rows(rows, require_identity_disjoint=True)

    def test_referenced_file_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "test" / "id_0123456789abcdef" / "a.png"
            _write_png(image, 8, 8, b"before")
            rows = discover_manifest_rows(root, "dataset", "license")
            _write_png(image, 8, 8, b"after")
            with self.assertRaisesRegex(DatasetManifestError, "hash mismatch"):
                validate_manifest_rows(rows, artifact_root=root)

    def test_snapshot_metadata_binds_source_and_manifest_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_png(root / "test" / "id_0123456789abcdef" / "a.png", 8, 8, b"a")
            rows = discover_manifest_rows(root, "dataset", "license")
            report = validate_manifest_rows(rows)
            metadata = build_snapshot_metadata(
                dataset_id="dataset",
                manifest_relative_uri="data/manifests/dataset.jsonl",
                report=report,
                source_archive_sha256="a" * 64,
                source_uri="https://example.test/dataset",
                source_retrieved_at="2026-08-22",
                license_id="license",
                created_at="2026-08-22T00:00:00Z",
            )
            self.assertEqual(metadata["manifest_sha256"], report.manifest_sha256)
            self.assertEqual(metadata["source_archive_sha256"], "a" * 64)
            json.dumps(metadata)

            with self.assertRaisesRegex(DatasetManifestError, "differs"):
                build_snapshot_metadata(
                    dataset_id="different-dataset",
                    manifest_relative_uri="data/manifests/dataset.jsonl",
                    report=report,
                    source_archive_sha256="a" * 64,
                    source_uri="https://example.test/dataset",
                    source_retrieved_at="2026-08-22",
                    license_id="license",
                )

    def test_cli_builds_manifest_and_snapshot_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "artifacts"
            _write_png(root / "test" / "id_0123456789abcdef" / "a.png", 8, 8, b"a")
            manifest = base / "manifest.jsonl"
            metadata = base / "snapshot.json"
            output = io.StringIO()
            with redirect_stdout(output):
                result = main(
                    [
                        "build",
                        "--artifact-root", str(root),
                        "--manifest-output", str(manifest),
                        "--metadata-output", str(metadata),
                        "--manifest-uri", "data/manifests/dataset.jsonl",
                        "--dataset-id", "dataset",
                        "--license-id", "license",
                        "--source-uri", "https://example.test/dataset",
                        "--source-retrieved-at", "2026-08-22",
                        "--source-archive-sha256", "a" * 64,
                    ]
                )
            self.assertEqual(result, 0)
            self.assertTrue(manifest.is_file())
            snapshot = json.loads(metadata.read_text(encoding="utf-8"))
            printed = json.loads(output.getvalue())
            self.assertEqual(snapshot["manifest_sha256"], printed["manifest_sha256"])


def _write_png(path: Path, width: int, height: int, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
    path.write_bytes(header + struct.pack(">II", width, height) + payload)


def _write_jpeg(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    app_segment = b"\xff\xe0\x00\x04\x00\x00"
    frame_data = b"\x08" + struct.pack(">HH", height, width) + b"\x01"
    frame_segment = b"\xff\xc0" + struct.pack(">H", len(frame_data) + 2) + frame_data
    path.write_bytes(b"\xff\xd8" + app_segment + frame_segment + b"\xff\xd9")


if __name__ == "__main__":
    unittest.main()
