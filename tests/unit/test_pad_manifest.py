import tempfile
import unittest
from pathlib import Path

from src.face_auth.evaluation.pad_manifest import (
    PADManifestError,
    PADManifestRow,
    load_pad_manifest,
    validate_pad_manifest,
)


def row(**changes):
    values = {
        "sample_id": "sample_00000001",
        "label": "bona_fide",
        "attack_species": "none",
        "relative_video_path": "live/sample.mp4",
        "subject_token": "subject_00000001",
        "session_token": "session_00000001",
        "device_token": "device_00000001",
        "split": "test",
    }
    values.update(changes)
    return PADManifestRow(**values)


class PADManifestTest(unittest.TestCase):
    def test_csv_loads_with_required_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pad.csv"
            path.write_text(
                ",".join(
                    [
                        "sample_id",
                        "label",
                        "attack_species",
                        "relative_video_path",
                        "subject_token",
                        "session_token",
                        "device_token",
                        "split",
                    ]
                )
                + "\n"
                + ",".join(row().__dict__.values())
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_pad_manifest(path), (row(),))

    def test_attack_requires_concrete_species(self):
        with self.assertRaisesRegex(PADManifestError, "concrete attack_species"):
            validate_pad_manifest([row(label="attack")])

    def test_parent_path_is_rejected(self):
        with self.assertRaisesRegex(PADManifestError, "safe relative"):
            validate_pad_manifest([row(relative_video_path="../secret.mp4")])

    def test_unknown_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pad.csv"
            path.write_text(
                ",".join((*row().__dict__.keys(), "typo_column"))
                + "\n"
                + ",".join((*row().__dict__.values(), "value"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                PADManifestError, "Unknown PAD manifest columns"
            ):
                load_pad_manifest(path)

    def test_duplicate_media_is_rejected(self):
        with self.assertRaisesRegex(PADManifestError, "Duplicate relative_video_path"):
            validate_pad_manifest([row(), row(sample_id="sample_00000002")])

    def test_subject_cannot_cross_calibration_and_test(self):
        with self.assertRaisesRegex(PADManifestError, "subject token crosses splits"):
            validate_pad_manifest(
                [
                    row(split="calibration"),
                    row(
                        sample_id="sample_00000002",
                        session_token="session_00000002",
                        relative_video_path="live/other.mp4",
                        split="test",
                    ),
                ]
            )


if __name__ == "__main__":
    unittest.main()
