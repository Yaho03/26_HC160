"""랜덤화 변환.

BPDA는 공격 루프에서 방어의 변환을 그대로 재현할 수 있기 때문에 통한다. 파라미터를
매번 무작위로 뽑으면 공격자가 고정 목표를 잡지 못한다.

07_DEFENSE_AND_DETECTION_SPEC.md 4절은 randomized method를 certificate 없이는
stochastic heuristic으로 표현하라고 요구한다. 여기서 certificate를 만들지 않는다.
"""

import unittest

import numpy as np
from PIL import Image

from src.verification.defenses.randomized_squeeze import (
    RandomizedTransformSpec,
    UnknownFamilyError,
    sample_transform,
    transform_families,
)


def _image(seed=0):
    pixels = np.random.default_rng(seed).integers(0, 256, (160, 160, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


class FamilyTest(unittest.TestCase):
    def test_families_cover_the_deployed_transforms(self):
        families = transform_families()
        for required in ("blur", "jpeg", "median"):
            self.assertIn(required, families)

    def test_unknown_family_is_rejected(self):
        with self.assertRaises(UnknownFamilyError):
            sample_transform("wishful", np.random.default_rng(0))


class SamplingTest(unittest.TestCase):
    def test_same_seed_gives_the_same_parameter(self):
        """재현 없이는 결과를 검증할 수 없다."""
        left = sample_transform("blur", np.random.default_rng(7))
        right = sample_transform("blur", np.random.default_rng(7))
        self.assertEqual(left.params, right.params)

    def test_different_draws_give_different_parameters(self):
        rng = np.random.default_rng(0)
        drawn = {sample_transform("blur", rng).params["radius"] for _ in range(20)}
        self.assertGreater(len(drawn), 1)

    def test_parameters_stay_inside_the_declared_range(self):
        rng = np.random.default_rng(0)
        for family, bounds in transform_families().items():
            for _ in range(30):
                spec = sample_transform(family, rng)
                value = next(iter(spec.params.values()))
                with self.subTest(family=family, value=value):
                    self.assertGreaterEqual(value, bounds[0])
                    self.assertLessEqual(value, bounds[1])

    def test_spec_records_family_and_params(self):
        spec = sample_transform("jpeg", np.random.default_rng(0))
        self.assertEqual(spec.family, "jpeg")
        self.assertIn("quality", spec.params)


class ApplyTest(unittest.TestCase):
    def test_applying_a_spec_changes_the_image(self):
        rng = np.random.default_rng(0)
        source = _image()
        for family in transform_families():
            with self.subTest(family=family):
                spec = sample_transform(family, rng)
                result = spec.apply(source)
                self.assertEqual(result.size, source.size)
                self.assertFalse(np.array_equal(np.asarray(result), np.asarray(source)))

    def test_same_spec_is_deterministic(self):
        spec = RandomizedTransformSpec("blur", {"radius": 0.9})
        source = _image()
        self.assertTrue(
            np.array_equal(np.asarray(spec.apply(source)), np.asarray(spec.apply(source)))
        )


if __name__ == "__main__":
    unittest.main()


class RandomizationAwareAttackTest(unittest.TestCase):
    """랜덤화를 아는 공격자와 모르는 공격자를 모두 평가해야 한다."""

    def _crop(self):
        return Image.fromarray(
            np.random.default_rng(0).integers(0, 256, (160, 160, 3), dtype=np.uint8)
        )

    def test_spec_batch_keeps_gradients_flowing(self):
        import torch

        from src.verification.defenses.bpda import bpda_spec_batch

        rng = np.random.default_rng(0)
        specs = [sample_transform(f, rng) for f in ("blur", "jpeg", "median")]
        pixels = np.random.default_rng(0).integers(0, 256, (1, 3, 160, 160)).astype(np.float32)
        source = torch.tensor((pixels - 127.5) / 128.0).requires_grad_(True)

        bpda_spec_batch(source, specs).sum().backward()

        self.assertIsNotNone(source.grad)
        self.assertGreater(float(source.grad.abs().sum()), 0.0)

    def test_attack_accepts_randomized_families(self):
        import torch

        from src.verification.defenses.adaptive_attack import (
            AdaptiveAttackConfig,
            run_eot_attack,
        )

        target = torch.nn.functional.normalize(torch.randn(1, 512), dim=1).squeeze(0)
        adversarial, _ = run_eot_attack(
            self._crop(), target, AdaptiveAttackConfig(steps=1),
            transforms=None, randomized_families=["blur", "jpeg"],
            rng=np.random.default_rng(0),
        )
        self.assertEqual(adversarial.size, (160, 160))

    def test_empty_spec_batch_is_rejected(self):
        import torch

        from src.verification.defenses.bpda import bpda_spec_batch

        with self.assertRaises(ValueError):
            bpda_spec_batch(torch.zeros(1, 3, 8, 8), [])
