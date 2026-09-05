"""장치 선택과 배치 처리.

계측과 공격 평가는 표본이 늘수록 시간이 선형으로 늘어난다. 팀원 표본이 들어오면
같은 평가를 다시 돌려야 하므로 낭비를 줄여 둔다.
"""

import unittest

import numpy as np
import torch
from PIL import Image

from src.verification.defenses.facenet_embed import select_device


class SelectDeviceTest(unittest.TestCase):
    def test_explicit_device_wins(self):
        self.assertEqual(select_device("cpu").type, "cpu")

    def test_returns_an_available_accelerator_or_cpu(self):
        device = select_device(None)
        self.assertIn(device.type, {"cuda", "mps", "cpu"})

    def test_prefers_mps_over_cpu_when_available(self):
        """Apple Silicon에서 CPU로 떨어지면 평가가 몇 배 느려진다."""
        if not torch.backends.mps.is_available():
            self.skipTest("MPS 없음")
        self.assertEqual(select_device(None).type, "mps")


class BatchedBpdaTest(unittest.TestCase):
    def _tensor(self):
        pixels = np.random.default_rng(0).integers(0, 256, (1, 3, 160, 160)).astype(np.float32)
        return torch.tensor((pixels - 127.5) / 128.0)

    def test_batch_matches_individual_transforms(self):
        """배치가 값을 바꾸면 안 된다. 속도만 달라져야 한다."""
        from src.verification.defenses.bpda import bpda_transform, bpda_transform_batch

        source = self._tensor()
        names = ["median3", "jpeg_q75", "median5"]

        batched = bpda_transform_batch(source, names)
        self.assertEqual(batched.shape[0], len(names))
        for index, name in enumerate(names):
            with self.subTest(name=name):
                single = bpda_transform(source, name)
                self.assertTrue(torch.allclose(batched[index : index + 1], single, atol=1e-5))

    def test_batch_keeps_gradients_flowing(self):
        from src.verification.defenses.bpda import bpda_transform_batch

        source = self._tensor().requires_grad_(True)
        bpda_transform_batch(source, ["median3", "jpeg_q75"]).sum().backward()

        self.assertIsNotNone(source.grad)
        self.assertGreater(float(source.grad.abs().sum()), 0.0)

    def test_empty_selection_is_rejected(self):
        from src.verification.defenses.bpda import bpda_transform_batch

        with self.assertRaises(ValueError):
            bpda_transform_batch(self._tensor(), [])


if __name__ == "__main__":
    unittest.main()
