import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.face_auth.domain.types import FramePacket
from src.face_auth.evaluation.pad_capture import (
    PADCaptureConfig,
    PADCaptureRecorder,
    relative_pad_video_path,
)
from src.face_auth.evaluation.pad_manifest import (
    PADManifestError,
    PADManifestRow,
    load_pad_manifest,
)


class FakeSource:
    def __init__(self, count):
        self.frames = [
            FramePacket(
                frame_id=index,
                captured_at_monotonic=float(index),
                image_bgr=np.full((12, 16, 3), index, dtype=np.uint8),
            )
            for index in range(count)
        ]
        self.read_count = 0
        self.closed = False

    def read(self):
        if self.read_count >= len(self.frames):
            return None
        frame = self.frames[self.read_count]
        self.read_count += 1
        return frame

    def close(self):
        self.closed = True


class FakeWriter:
    def __init__(self, path, _codec, _fps, _size, *, on_release=None):
        self.path = Path(path)
        self.frames = []
        self.on_release = on_release

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.path.write_bytes(b"fake-mp4:" + str(len(self.frames)).encode("ascii"))
        if self.on_release:
            self.on_release()


def writer_factory(on_release=None):
    return lambda *args: FakeWriter(*args, on_release=on_release)


def row(**changes):
    values = {
        "sample_id": "sample_00000001",
        "label": "bona_fide",
        "attack_species": "none",
        "relative_video_path": "placeholder.mp4",
        "subject_token": "subject_00000001",
        "session_token": "session_00000001",
        "device_token": "device_00000001",
        "split": "calibration",
    }
    values.update(changes)
    provisional = PADManifestRow(**values)
    return PADManifestRow(
        **{
            **provisional.__dict__,
            "relative_video_path": relative_pad_video_path(provisional),
        }
    )


def config(root, **changes):
    values = {
        "artifact_root": root,
        "manifest_path": root / "manifest.csv",
        "fps": 5.0,
        "frame_count": 5,
        "min_frames": 3,
        "width": 32,
        "height": 24,
    }
    values.update(changes)
    return PADCaptureConfig(**values)


class PADCaptureRecorderTest(unittest.TestCase):
    def test_success_registers_video_and_manifest_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = FakeSource(5)
            sample = row()
            receipt = PADCaptureRecorder(
                source, writer_factory=writer_factory()
            ).capture(sample, config(root))

            self.assertEqual(receipt.frame_count, 5)
            self.assertGreater(receipt.video_bytes, 0)
            self.assertTrue((root / sample.relative_video_path).is_file())
            self.assertEqual(load_pad_manifest(root / "manifest.csv"), (sample,))
            self.assertTrue(source.closed)
            self.assertFalse((root / ".pad-capture.lock").exists())

    def test_too_few_frames_leave_no_video_or_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = row()
            with self.assertRaisesRegex(RuntimeError, "too few frames"):
                PADCaptureRecorder(
                    FakeSource(2), writer_factory=writer_factory()
                ).capture(sample, config(root))
            self.assertFalse((root / sample.relative_video_path).exists())
            self.assertFalse((root / "manifest.csv").exists())
            self.assertFalse(any(root.rglob("*.partial.mp4")))
            self.assertFalse((root / ".pad-capture.lock").exists())

    def test_subject_split_leakage_is_rejected_before_frames_are_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            PADCaptureRecorder(FakeSource(5), writer_factory=writer_factory()).capture(
                row(), config(root)
            )
            source = FakeSource(5)
            test_row = row(
                sample_id="sample_00000002",
                session_token="session_00000002",
                split="test",
            )
            with self.assertRaisesRegex(PADManifestError, "crosses splits"):
                PADCaptureRecorder(source, writer_factory=writer_factory()).capture(
                    test_row, config(root)
                )
            self.assertEqual(source.read_count, 0)
            self.assertTrue(source.closed)

    def test_manifest_change_during_capture_removes_unregistered_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"

            def mutate_manifest():
                manifest.write_text("changed", encoding="utf-8")

            sample = row()
            with self.assertRaisesRegex(RuntimeError, "changed while capture"):
                PADCaptureRecorder(
                    FakeSource(5), writer_factory=writer_factory(mutate_manifest)
                ).capture(sample, config(root))
            self.assertFalse((root / sample.relative_video_path).exists())

    def test_manifest_must_stay_inside_dataset_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "inside artifact root"):
                PADCaptureRecorder(
                    FakeSource(5), writer_factory=writer_factory()
                ).capture(
                    row(),
                    config(root, manifest_path=Path(directory) / "outside.csv"),
                )

    def test_existing_dataset_lock_rejects_concurrent_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / ".pad-capture.lock"
            lock.write_text("pid=other\n", encoding="ascii")
            source = FakeSource(5)
            with self.assertRaisesRegex(RuntimeError, "owns the dataset lock"):
                PADCaptureRecorder(source, writer_factory=writer_factory()).capture(
                    row(), config(root)
                )
            self.assertEqual(source.read_count, 0)
            self.assertTrue(source.closed)
            self.assertTrue(lock.exists())


if __name__ == "__main__":
    unittest.main()
