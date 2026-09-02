import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.verification.defenses.probe_log import (
    PROBE_COLUMNS,
    ProbeWriter,
    OpaqueIdError,
    SidecarContentError,
    write_session_sidecar,
)
from src.verification.defenses.squeeze_probe import (
    TRANSFORM_ORDER,
    ProbeReading,
    TransformReading,
)


def _reading(offset=0.0):
    readings = tuple(
        TransformReading(
            transform=name,
            cos_orig_enroll=0.80 + offset,
            cos_transformed_enroll=0.70 + offset,
            cos_orig_transformed=0.95 + offset,
        )
        for name in TRANSFORM_ORDER
    )
    return ProbeReading(readings=readings, embed_ms=12.5)


class ProbeWriterTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name) / "probe.csv"

    def tearDown(self):
        self._dir.cleanup()

    def test_header_matches_declared_schema(self):
        with ProbeWriter(self.path, session_id="s1", subject_id="p01") as writer:
            writer.write_sample(
                sample_id="f000_clean",
                frame_idx=0,
                frame_ts_ms=0.0,
                dropped_frames=0,
                label="clean",
                reading=_reading(),
            )

        with self.path.open() as handle:
            header = next(csv.reader(handle))
        self.assertEqual(tuple(header), PROBE_COLUMNS)

    def test_one_sample_writes_one_row_per_transform(self):
        with ProbeWriter(self.path, session_id="s1", subject_id="p01") as writer:
            writer.write_sample(
                sample_id="f000_clean",
                frame_idx=0,
                frame_ts_ms=0.0,
                dropped_frames=0,
                label="clean",
                reading=_reading(),
            )

        rows = list(csv.DictReader(self.path.open()))
        self.assertEqual(len(rows), len(TRANSFORM_ORDER))
        self.assertEqual({row["sample_id"] for row in rows}, {"f000_clean"})
        self.assertEqual(
            [row["transform"] for row in rows], list(TRANSFORM_ORDER)
        )

    def test_clean_and_adversarial_rows_join_on_frame_idx(self):
        with ProbeWriter(self.path, session_id="s1", subject_id="p01") as writer:
            writer.write_sample(
                sample_id="f007_clean",
                frame_idx=7,
                frame_ts_ms=470.0,
                dropped_frames=1,
                label="clean",
                reading=_reading(),
            )
            writer.write_sample(
                sample_id="f007_adv",
                frame_idx=7,
                frame_ts_ms=470.0,
                dropped_frames=1,
                label="adversarial",
                reading=_reading(offset=-0.30),
            )

        rows = list(csv.DictReader(self.path.open()))
        frame_seven = [row for row in rows if row["frame_idx"] == "7"]
        self.assertEqual(len(frame_seven), 2 * len(TRANSFORM_ORDER))
        self.assertEqual(
            {row["label"] for row in frame_seven}, {"clean", "adversarial"}
        )

    def test_rejects_label_outside_the_declared_set(self):
        with ProbeWriter(self.path, session_id="s1", subject_id="p01") as writer:
            with self.assertRaises(ValueError):
                writer.write_sample(
                    sample_id="f000_x",
                    frame_idx=0,
                    frame_ts_ms=0.0,
                    dropped_frames=0,
                    label="unknown",
                    reading=_reading(),
                )

    def test_rejects_subject_id_that_leaks_identity(self):
        for leaky in ("/Users/choi/face.jpg", "choi@example.com", "최환석"):
            with self.subTest(subject_id=leaky):
                with self.assertRaises(OpaqueIdError):
                    ProbeWriter(self.path, session_id="s1", subject_id=leaky)


class SessionSidecarTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name) / "session.json"

    def tearDown(self):
        self._dir.cleanup()

    def _meta(self, **overrides):
        meta = {
            "session_id": "s1",
            "subject_id": "p01",
            "created_at": "2026-09-02T00:00:00+00:00",
            "git_commit": "abc1234",
            "camera": {"index": 0, "width": 1280, "height": 720, "fps_nominal": 30.0},
            "model": {"name": "InceptionResnetV1", "pretrained": "vggface2"},
            "transforms": {"jpeg_q75": {"quality": 75}},
            "attack": {"kind": "pgd", "epsilon": 0.03, "steps": 40, "every": 10},
            "counters": {"frames_captured": 300, "samples_clean": 300},
        }
        meta.update(overrides)
        return meta

    def test_sidecar_round_trips_provenance(self):
        write_session_sidecar(self.path, self._meta())
        stored = json.loads(self.path.read_text())

        for key in ("session_id", "git_commit", "camera", "model", "transforms", "attack"):
            self.assertIn(key, stored)

    def test_sidecar_rejects_absolute_paths(self):
        meta = self._meta(notes="/Users/choi/Desktop/capture.mp4")
        with self.assertRaises(SidecarContentError):
            write_session_sidecar(self.path, meta)

    def test_sidecar_rejects_embedding_vectors(self):
        meta = self._meta(enroll_embedding=[0.1, 0.2, 0.3])
        with self.assertRaises(SidecarContentError):
            write_session_sidecar(self.path, meta)


if __name__ == "__main__":
    unittest.main()
