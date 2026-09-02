"""capture_session 루프 자체를 가짜 카메라로 돌린다.

순수 헬퍼만 테스트하면 루프 안의 이름 오류나 순서 오류를 놓친다. 실제로 그런
NameError가 한 번 통과한 적이 있다.
"""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from src.verification.defenses.probe_log import PROBE_COLUMNS
from src.verification.defenses.squeeze_probe import TRANSFORM_ORDER


class FakeCapture:
    def __init__(self, frames: int = 50) -> None:
        self._remaining = frames
        self._rng = np.random.default_rng(0)

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def get(self, prop):
        return 0.0

    def read(self):
        if self._remaining <= 0:
            return False, None
        self._remaining -= 1
        return True, self._rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)

    def release(self):
        pass


def _fake_crop(frame, device=None):
    rng = np.random.default_rng(int(frame.sum()) % (2**32))
    pixels = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
    return Image.fromarray(pixels), [10, 10, 100, 100]


class StubEmbedder:
    def __init__(self, device=None):
        self.device = device

    def embed_batch(self, images):
        rng = np.random.default_rng(len(images))
        vectors = rng.normal(size=(len(images), 8))
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class CaptureSessionLoopTest(unittest.TestCase):
    def _run(self, frames, attack_every, camera_frames=200, attack_kinds=None):
        import src.verification.defenses.probe_capture as module

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(module.cv2, "VideoCapture", lambda *a, **k: FakeCapture(camera_frames)), \
                 mock.patch.object(module, "detect_and_crop", _fake_crop), \
                 mock.patch.object(module, "FaceNetBatchEmbedder", StubEmbedder), \
                 mock.patch.object(module, "get_embedding", lambda img, device=None: __import__("torch").ones(8) / (8 ** 0.5)), \
                 mock.patch.object(module, "run_attack", lambda kind, crop, emb, cfg, **kw: (crop, 0.9)), \
                 mock.patch.object(module, "_weights_sha256", lambda: "0" * 64):
                # 계측 진행 출력이 CI 로그를 덮지 않게 한다.
                with contextlib.redirect_stdout(io.StringIO()):
                    csv_path, sidecar_path = module.capture_session(
                        subject_id="p01",
                        frames=frames,
                        attack_every=attack_every,
                        out_dir=Path(directory),
                        camera_index=0,
                        enroll_img_path=None,
                        epsilon=0.03,
                        steps=1,
                        step_size=0.002,
                        no_preview=True,
                        attack_kinds=attack_kinds,
                    )
                rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        return rows, sidecar

    def test_completed_session_writes_expected_row_count(self):
        rows, sidecar = self._run(frames=10, attack_every=5)

        self.assertEqual(
            {row["attack_kind"] for row in rows if row["label"] == "adversarial"}, {"pgd"}
        )
        self.assertEqual(
            {row["attack_kind"] for row in rows if row["label"] == "clean"}, {""}
        )

        clean = {row["sample_id"] for row in rows if row["label"] == "clean"}
        adversarial = {row["sample_id"] for row in rows if row["label"] == "adversarial"}

        self.assertEqual(len(clean), 10)
        self.assertEqual(len(adversarial), 2)
        self.assertEqual(len(rows), (10 + 2) * len(TRANSFORM_ORDER))
        self.assertTrue(sidecar["completed"])
        self.assertIsNone(sidecar["interrupted_by"])

    def test_schema_and_headroom_are_recorded(self):
        rows, sidecar = self._run(frames=6, attack_every=0)

        self.assertEqual(tuple(rows[0]), PROBE_COLUMNS)
        self.assertIsNotNone(sidecar["jpeg_headroom_q75"])
        self.assertEqual(sidecar["counters"]["samples_adversarial"], 0)

    def test_attack_every_zero_records_clean_only(self):
        rows, _ = self._run(frames=6, attack_every=0)
        self.assertEqual({row["label"] for row in rows}, {"clean"})

    def test_multiple_attack_kinds_rotate_within_one_session(self):
        """촬영은 사람 시간이 드니 한 세션에서 여러 공격을 모은다."""
        rows, sidecar = self._run(
            frames=12, attack_every=2, attack_kinds=["pgd", "fgsm"]
        )
        kinds = {row["attack_kind"] for row in rows if row["label"] == "adversarial"}

        self.assertEqual(kinds, {"pgd", "fgsm"})
        self.assertEqual(set(sidecar["attack"]["kinds"]), {"pgd", "fgsm"})
        self.assertGreater(sidecar["attack"]["counts"]["fgsm"], 0)

    def test_camera_exhaustion_still_writes_a_sidecar(self):
        """카메라가 먼저 끊겨도 provenance 없는 CSV를 남기면 안 된다."""
        rows, sidecar = self._run(frames=500, attack_every=0, camera_frames=20)

        self.assertFalse(sidecar["completed"])
        self.assertEqual(sidecar["interrupted_by"], "read_failure")
        self.assertGreater(len(rows), 0)


if __name__ == "__main__":
    unittest.main()
