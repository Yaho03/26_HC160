"""face_auth 게이트의 랜덤화 변환.

연구 트랙 실측에서 고정 변환은 BPDA에 탐지율 0%까지 뚫렸고 파라미터를 무작위로
뽑으니 84%로 회복됐다. 근거는 docs/experiments/EXP-DET-001-camera-squeeze-probe.md
12절과 16절이다.

이번 작업은 이식이며 face_auth 경로에서의 탐지율은 측정하지 않았다. 따라서 랜덤화의
기본값은 꺼짐이고, 끄면 기존 동작과 완전히 같아야 한다.
"""

import unittest

import numpy as np
from PIL import Image

from src.face_auth.domain.types import GateResult, GateStatus
from src.face_auth.inference.adversarial_detector import (
    AdversarialDetectorConfig,
    TemplateShiftDetector,
    TemplateShiftDetectorConfig,
    TransformConsistencyDetector,
)
from src.face_auth.inference.feature_squeeze import (
    FeatureSqueezeConfig,
    FeatureSqueezeInspector,
    _transforms,
)


def _unit(*components):
    vector = np.asarray(components, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _crop(seed=0):
    pixels = np.random.default_rng(seed).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


class RecordingEmbedder:
    def __init__(self, vector=None):
        self.vector = vector if vector is not None else _unit(1, 0)
        self.batches = []

    def embed(self, images):
        self.batches.append(list(images))
        return [self.vector for _ in images]


def _inspector(embedder, config=None, rng=None, template=None):
    template_detector = (
        TemplateShiftDetector(
            TemplateShiftDetectorConfig(max_template_shift=0.5, threshold_version="t"),
            template,
        )
        if template is not None
        else None
    )
    return FeatureSqueezeInspector(
        embedder,
        TransformConsistencyDetector(
            AdversarialDetectorConfig(max_cosine_distance=0.5, threshold_version="c")
        ),
        config=config,
        template_detector=template_detector,
        rng=rng,
    )


class DefaultsTest(unittest.TestCase):
    """검증되지 않은 변경을 기본 동작으로 만들면 안 된다."""

    def test_randomization_is_off_by_default(self):
        self.assertFalse(FeatureSqueezeConfig().randomized)

    def test_default_families_match_the_shipped_transforms(self):
        self.assertEqual(FeatureSqueezeConfig().families, ("jpeg", "bit", "blur"))

    def test_default_preset_is_explicit(self):
        """모듈 기본값에 기대면 연구 트랙이 범위를 바꿀 때 인증이 조용히 바뀐다."""
        self.assertEqual(FeatureSqueezeConfig().range_preset, "narrow")


class FixedModeUnchangedTest(unittest.TestCase):
    """랜덤화를 끄면 기존 동작과 완전히 같아야 한다."""

    def test_transforms_are_byte_identical_to_the_previous_behaviour(self):
        crop = _crop()
        config = FeatureSqueezeConfig()
        produced = _transforms(crop, config)

        from io import BytesIO

        from PIL import ImageFilter

        buffer = BytesIO()
        crop.convert("RGB").save(buffer, format="JPEG", quality=75)
        buffer.seek(0)
        expected_jpeg = Image.open(buffer).convert("RGB").copy()

        levels = float((1 << 5) - 1)
        array = np.asarray(crop.convert("RGB"), dtype=np.float32) / 255.0
        expected_bits = Image.fromarray(
            np.clip(np.rint(array * levels) / levels * 255.0, 0, 255).astype(np.uint8)
        )
        expected_blur = crop.convert("RGB").filter(ImageFilter.GaussianBlur(0.8))

        for produced_image, expected in zip(
            produced, (expected_jpeg, expected_bits, expected_blur)
        ):
            self.assertTrue(
                np.array_equal(np.asarray(produced_image), np.asarray(expected))
            )

    def test_fixed_mode_leaves_no_transform_evidence(self):
        """고정 변환은 config에 이미 적혀 있으므로 evidence가 필요 없다."""
        results = _inspector(RecordingEmbedder()).evaluate([_crop()], [_unit(1, 0)])
        self.assertEqual(results[0].evidence, ())

    def test_fixed_mode_ignores_the_rng(self):
        embedder_a, embedder_b = RecordingEmbedder(), RecordingEmbedder()
        _inspector(embedder_a, rng=np.random.default_rng(1)).evaluate(
            [_crop()], [_unit(1, 0)]
        )
        _inspector(embedder_b, rng=np.random.default_rng(2)).evaluate(
            [_crop()], [_unit(1, 0)]
        )
        for left, right in zip(embedder_a.batches[0], embedder_b.batches[0]):
            self.assertTrue(np.array_equal(np.asarray(left), np.asarray(right)))


class RandomizedModeTest(unittest.TestCase):
    def _config(self, **overrides):
        base = {"randomized": True}
        base.update(overrides)
        return FeatureSqueezeConfig(**base)

    def test_same_seed_produces_the_same_transforms(self):
        embedder_a, embedder_b = RecordingEmbedder(), RecordingEmbedder()
        for embedder in (embedder_a, embedder_b):
            _inspector(
                embedder, config=self._config(), rng=np.random.default_rng(42)
            ).evaluate([_crop()], [_unit(1, 0)])

        for left, right in zip(embedder_a.batches[0], embedder_b.batches[0]):
            self.assertTrue(np.array_equal(np.asarray(left), np.asarray(right)))

    def test_different_seeds_produce_different_transforms(self):
        embedder_a, embedder_b = RecordingEmbedder(), RecordingEmbedder()
        _inspector(embedder_a, config=self._config(), rng=np.random.default_rng(1)).evaluate(
            [_crop()], [_unit(1, 0)]
        )
        _inspector(embedder_b, config=self._config(), rng=np.random.default_rng(2)).evaluate(
            [_crop()], [_unit(1, 0)]
        )

        differs = any(
            not np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(embedder_a.batches[0], embedder_b.batches[0])
        )
        self.assertTrue(differs)

    def test_parameters_are_redrawn_for_each_frame(self):
        """한 번 뽑아 모든 프레임에 재사용하면 공격자가 한 번만 맞히면 된다."""
        embedder = RecordingEmbedder()
        _inspector(embedder, config=self._config(), rng=np.random.default_rng(0)).evaluate(
            [_crop(0), _crop(1)], [_unit(1, 0), _unit(1, 0)]
        )

        self.assertEqual(len(embedder.batches), 2)

    def test_evidence_records_the_sampled_parameters(self):
        """07 6절이 transformation별 evidence를 요구한다."""
        results = _inspector(
            RecordingEmbedder(), config=self._config(), rng=np.random.default_rng(0)
        ).evaluate([_crop()], [_unit(1, 0)])

        evidence = results[0].evidence
        self.assertEqual(len(evidence), 3)
        self.assertTrue(all("=" in item for item in evidence))
        self.assertEqual({item.split(":")[0] for item in evidence}, {"jpeg", "bit", "blur"})

    def test_both_gates_share_one_set_of_transforms(self):
        """게이트별로 따로 뽑으면 embedding forward가 두 배가 된다."""
        embedder = RecordingEmbedder()
        results = _inspector(
            embedder, config=self._config(), rng=np.random.default_rng(0),
            template=_unit(1, 0),
        ).evaluate([_crop()], [_unit(1, 0)])

        self.assertEqual(len(results), 2)
        self.assertEqual(len(embedder.batches), 1)

    def test_both_gates_record_the_same_evidence(self):
        results = _inspector(
            RecordingEmbedder(), config=self._config(), rng=np.random.default_rng(0),
            template=_unit(1, 0),
        ).evaluate([_crop()], [_unit(1, 0)])

        self.assertEqual(results[0].evidence, results[1].evidence)

    def test_explicit_preset_is_used(self):
        """모듈 기본값이 아니라 config가 범위를 정한다."""
        from src.verification.defenses.randomized_squeeze import RANGE_PRESETS

        results = _inspector(
            RecordingEmbedder(),
            config=self._config(range_preset="wide", families=("blur",)),
            rng=np.random.default_rng(0),
        ).evaluate([_crop()], [_unit(1, 0)])

        radius = float(results[0].evidence[0].split("=")[1])
        low, high = RANGE_PRESETS["wide"]["blur"]
        self.assertGreaterEqual(radius, low)
        self.assertLessEqual(radius, high)

    def test_unknown_family_is_rejected(self):
        from src.verification.defenses.randomized_squeeze import UnknownFamilyError

        with self.assertRaises(UnknownFamilyError):
            _inspector(
                RecordingEmbedder(),
                config=self._config(families=("wishful",)),
                rng=np.random.default_rng(0),
            ).evaluate([_crop()], [_unit(1, 0)])


class GateResultEvidenceTest(unittest.TestCase):
    def test_evidence_defaults_to_empty(self):
        self.assertEqual(GateResult("g", GateStatus.PASS).evidence, ())

    def test_evidence_is_hashable(self):
        """GateResult는 frozen dataclass다. 해시 가능해야 한다."""
        hash(GateResult("g", GateStatus.PASS, evidence=("blur:radius=0.8",)))


if __name__ == "__main__":
    unittest.main()
