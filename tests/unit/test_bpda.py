"""BPDA: 미분 불가 변환의 backward를 항등함수로 근사한다.

EOT는 미분 가능한 blur만 최적화할 수 있어, detector 점수의 대부분을 차지하는
JPEG와 median 항을 건드리지 못했다. BPDA는 forward에 진짜 변환을 쓰고 backward만
항등으로 근사해 그 항들도 최적화 대상으로 만든다.

이것이 07_DEFENSE_AND_DETECTION_SPEC.md 7.1절이 지목한 우회 기법이다.
"""

import unittest

import numpy as np
import torch

from src.verification.defenses.bpda import bpda_transform, supported_transforms


def _tensor(seed=0):
    pixels = np.random.default_rng(seed).integers(0, 256, (1, 3, 160, 160)).astype(np.float32)
    return torch.tensor((pixels - 127.5) / 128.0)


class SupportTest(unittest.TestCase):
    def test_covers_the_non_differentiable_transforms(self):
        names = supported_transforms()
        for required in ("jpeg_q75", "jpeg_q30", "median3", "median5"):
            self.assertIn(required, names)

    def test_unknown_transform_is_rejected(self):
        with self.assertRaises(KeyError):
            bpda_transform(_tensor(), "not_a_transform")


class ForwardTest(unittest.TestCase):
    def test_forward_matches_the_real_transform(self):
        """근사는 backward에만 적용한다. forward는 방어가 실제로 하는 연산이어야 한다."""
        from src.verification.defenses.bpda import _to_pil, _to_tensor
        from src.verification.defenses.squeeze_probe import TRANSFORMS

        source = _tensor()
        for name in ("median3", "jpeg_q75"):
            with self.subTest(name=name):
                through_bpda = bpda_transform(source, name)
                direct = _to_tensor(TRANSFORMS[name](_to_pil(source)), source.device)
                self.assertTrue(torch.allclose(through_bpda, direct, atol=1e-5))

    def test_transform_actually_changes_the_input(self):
        source = _tensor()
        for name in ("median3", "jpeg_q30"):
            with self.subTest(name=name):
                self.assertFalse(torch.allclose(bpda_transform(source, name), source))


class BackwardTest(unittest.TestCase):
    def test_gradient_flows_through_a_non_differentiable_transform(self):
        """이것이 BPDA의 요점이다. 근사가 없으면 gradient가 끊긴다."""
        source = _tensor().requires_grad_(True)
        output = bpda_transform(source, "median3")
        output.sum().backward()

        self.assertIsNotNone(source.grad)
        self.assertGreater(float(source.grad.abs().sum()), 0.0)

    def test_backward_is_the_identity_approximation(self):
        """sum()의 gradient는 항등 근사에서 모두 1이 된다."""
        source = _tensor().requires_grad_(True)
        bpda_transform(source, "jpeg_q75").sum().backward()

        self.assertTrue(torch.allclose(source.grad, torch.ones_like(source.grad)))

    def test_differentiable_transform_needs_no_approximation(self):
        """blur는 EOT가 직접 다룰 수 있으므로 BPDA 대상이 아니다."""
        self.assertNotIn("blur0.8", supported_transforms())


if __name__ == "__main__":
    unittest.main()
