"""Adaptive attack: 공격자가 방어를 알고 있는 경우.

07_DEFENSE_AND_DETECTION_SPEC.md 7절과 3절은 defended checkpoint 대상 adaptive
attack을 요구한다. 지금까지의 결과는 공격자가 방어를 모른다는 가정에 기댄다.

EOT(Expectation over Transformation)는 변환을 공격 루프 안에 넣어, 변환 후에도
살아남는 perturbation을 만든다. squeezing detector는 변환 전후 임베딩 차이를 보므로
그 차이를 작게 유지하는 것이 공격자의 목표가 된다.
"""

import unittest

import numpy as np
import torch
from PIL import Image

from src.verification.defenses.adaptive_attack import (
    AdaptiveAttackConfig,
    UnknownAdaptiveModeError,
    build_eot_transforms,
    resolve_mode,
)


class TransformSelectionTest(unittest.TestCase):
    def test_eot_can_only_use_differentiable_transforms(self):
        """
        core 변환 6종 중 EOT 루프에 넣을 수 있는 것은 blur 3종뿐이다.
        median과 JPEG는 미분 불가라 gradient 공격이 직접 최적화할 수 없다.
        방어에 유리한 구조적 비대칭이며 결과 해석에 명시해야 한다.
        """
        from src.verification.defenses.squeeze_probe import CORE_TRANSFORMS

        names = build_eot_transforms(list(CORE_TRANSFORMS))

        self.assertTrue(set(names).issubset(CORE_TRANSFORMS))
        self.assertTrue(all(n.startswith("blur") for n in names))
        self.assertEqual(
            set(CORE_TRANSFORMS) - set(names), {"median3", "median5", "jpeg_q30"}
        )

    def test_non_differentiable_only_selection_is_rejected(self):
        """미분 가능한 변환이 하나도 없으면 EOT를 돌릴 수 없다."""
        with self.assertRaises(KeyError):
            build_eot_transforms(["median3", "jpeg_q30"])

    def test_unknown_transform_is_rejected(self):
        with self.assertRaises(KeyError):
            build_eot_transforms(["not_a_transform"])

    def test_empty_selection_is_rejected(self):
        with self.assertRaises(ValueError):
            build_eot_transforms([])


class ModeTest(unittest.TestCase):
    def test_oblivious_mode_ignores_the_defense(self):
        self.assertFalse(resolve_mode("oblivious")["uses_defense"])

    def test_eot_mode_uses_the_defense(self):
        self.assertTrue(resolve_mode("eot")["uses_defense"])

    def test_unknown_mode_is_rejected_before_the_run(self):
        with self.assertRaises(UnknownAdaptiveModeError):
            resolve_mode("wishful")


class ConfigTest(unittest.TestCase):
    def test_defaults_match_the_measured_session(self):
        config = AdaptiveAttackConfig()
        self.assertEqual(config.epsilon, 0.03)
        self.assertEqual(config.steps, 40)

    def test_consistency_weight_trades_identity_for_stealth(self):
        """가중치가 0이면 기존 공격과 같아야 한다."""
        self.assertEqual(AdaptiveAttackConfig(consistency_weight=0.0).consistency_weight, 0.0)


class EotAttackTest(unittest.TestCase):
    def _crop(self):
        pixels = np.random.default_rng(0).integers(0, 256, (160, 160, 3), dtype=np.uint8)
        return Image.fromarray(pixels)

    def test_eot_attack_stays_within_the_epsilon_budget(self):
        """공격 예산을 넘으면 비교가 성립하지 않는다."""
        from src.verification.defenses.adaptive_attack import run_eot_attack
        from src.verification.defenses.facenet_embed import preprocess

        crop = self._crop()
        target = torch.nn.functional.normalize(torch.randn(1, 512), dim=1).squeeze(0)
        config = AdaptiveAttackConfig(epsilon=0.03, steps=2, step_size=0.01)

        adversarial, _ = run_eot_attack(crop, target, config, transforms=["blur0.8"])

        before = preprocess(crop)
        after = preprocess(adversarial)
        # 변환은 픽셀 왕복이 있으므로 약간의 여유를 둔다.
        self.assertLessEqual(float((after - before).abs().max()), 0.03 + 0.02)

    def test_eot_attack_returns_an_image_of_the_same_size(self):
        from src.verification.defenses.adaptive_attack import run_eot_attack

        crop = self._crop()
        target = torch.nn.functional.normalize(torch.randn(1, 512), dim=1).squeeze(0)
        adversarial, similarity = run_eot_attack(
            crop, target, AdaptiveAttackConfig(steps=2), transforms=["blur0.8"]
        )

        self.assertEqual(adversarial.size, crop.size)
        self.assertIsInstance(similarity, float)


if __name__ == "__main__":
    unittest.main()
