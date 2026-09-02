"""공격 종류 다양화.

단일 공격 종류로 산출한 임계값은 그 공격의 지문을 외운 것과 구별되지 않는다.
촬영은 사람 시간이 들어 되돌리기 비싸므로 한 세션에서 여러 공격을 함께 모은다.
"""

import unittest

import numpy as np
import torch
from PIL import Image

from src.verification.defenses.probe_attacks import (
    ATTACK_KINDS,
    AttackConfig,
    UnknownAttackError,
    attack_for_index,
    build_attack_params,
)


class AttackRotationTest(unittest.TestCase):
    def test_single_kind_always_returns_that_kind(self):
        kinds = [attack_for_index(i, ["pgd"]) for i in range(5)]
        self.assertEqual(set(kinds), {"pgd"})

    def test_multiple_kinds_rotate_evenly(self):
        kinds = [attack_for_index(i, ["pgd", "fgsm"]) for i in range(6)]
        self.assertEqual(kinds, ["pgd", "fgsm", "pgd", "fgsm", "pgd", "fgsm"])

    def test_rotation_covers_every_kind_within_one_cycle(self):
        kinds = ["pgd", "fgsm", "pgd_low_eps"]
        seen = {attack_for_index(i, kinds) for i in range(len(kinds))}
        self.assertEqual(seen, set(kinds))

    def test_unknown_kind_is_rejected_before_capture_starts(self):
        """촬영을 다 하고 나서 실패하면 사람 시간을 버린다."""
        with self.assertRaises(UnknownAttackError):
            build_attack_params(["pgd", "does_not_exist"], AttackConfig())


class AttackParamsTest(unittest.TestCase):
    def test_every_declared_kind_has_params(self):
        params = build_attack_params(list(ATTACK_KINDS), AttackConfig())
        self.assertEqual(set(params), set(ATTACK_KINDS))

    def test_fgsm_is_a_single_step(self):
        params = build_attack_params(["fgsm"], AttackConfig())
        self.assertEqual(params["fgsm"]["steps"], 1)

    def test_pgd_low_eps_is_weaker_than_pgd(self):
        params = build_attack_params(["pgd", "pgd_low_eps"], AttackConfig())
        self.assertLess(params["pgd_low_eps"]["epsilon"], params["pgd"]["epsilon"])

    def test_params_record_the_configured_epsilon(self):
        params = build_attack_params(["pgd"], AttackConfig(epsilon=0.05))
        self.assertEqual(params["pgd"]["epsilon"], 0.05)


class AttackExecutionTest(unittest.TestCase):
    def _crop(self):
        pixels = np.random.default_rng(0).integers(0, 256, (160, 160, 3), dtype=np.uint8)
        return Image.fromarray(pixels)

    def test_each_kind_produces_an_image_of_the_same_size(self):
        from src.verification.defenses.probe_attacks import run_attack

        target = torch.nn.functional.normalize(torch.randn(1, 512), dim=1).squeeze(0)
        crop = self._crop()
        calls = []

        def fake_pgd(source_img, target_emb, epsilon, n_steps, step_size, device=None):
            calls.append((epsilon, n_steps, step_size))
            return source_img, 0.9

        for kind in ATTACK_KINDS:
            with self.subTest(kind=kind):
                image, _ = run_attack(
                    kind, crop, target, AttackConfig(), generator=fake_pgd
                )
                self.assertEqual(image.size, crop.size)

        self.assertEqual(len(calls), len(ATTACK_KINDS))

    def test_fgsm_uses_one_step_with_full_epsilon(self):
        from src.verification.defenses.probe_attacks import run_attack

        target = torch.nn.functional.normalize(torch.randn(1, 512), dim=1).squeeze(0)
        seen = {}

        def fake_pgd(source_img, target_emb, epsilon, n_steps, step_size, device=None):
            seen.update(epsilon=epsilon, n_steps=n_steps, step_size=step_size)
            return source_img, 0.9

        run_attack("fgsm", self._crop(), target, AttackConfig(epsilon=0.03), generator=fake_pgd)

        self.assertEqual(seen["n_steps"], 1)
        self.assertEqual(seen["step_size"], 0.03)


if __name__ == "__main__":
    unittest.main()
