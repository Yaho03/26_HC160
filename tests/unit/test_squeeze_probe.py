import unittest

import numpy as np
from PIL import Image

from src.verification.defenses.squeeze_probe import (
    BASELINE_TRANSFORMS,
    CORE_TRANSFORMS,
    TRANSFORM_ORDER,
    TRANSFORM_PARAMS,
    TRANSFORMS,
    jpeg_headroom,
    probe_crop,
    self_consistency,
    template_shift,
)


class StubEmbedder:
    """Returns preset unit vectors in call order and records the batch it saw."""

    def __init__(self, vectors):
        self._vectors = np.asarray(vectors, dtype=np.float64)
        self.batches = []

    def embed_batch(self, images):
        self.batches.append(list(images))
        return self._vectors[: len(images)]


def _unit(*components):
    vector = np.asarray(components, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _crop():
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(160, 160, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


class SqueezeProbeTest(unittest.TestCase):
    def test_one_reading_per_transform_in_declared_order(self):
        embedder = StubEmbedder([_unit(1, 0)] * (1 + len(TRANSFORM_ORDER)))
        reading = probe_crop(_crop(), _unit(1, 0), embedder)

        self.assertEqual(
            [item.transform for item in reading.readings], list(TRANSFORM_ORDER)
        )

    def test_original_and_transforms_embed_in_a_single_batch(self):
        embedder = StubEmbedder([_unit(1, 0)] * (1 + len(TRANSFORM_ORDER)))
        probe_crop(_crop(), _unit(1, 0), embedder)

        self.assertEqual(len(embedder.batches), 1)
        self.assertEqual(len(embedder.batches[0]), 1 + len(TRANSFORM_ORDER))

    def test_raw_cosines_use_original_transformed_and_enroll(self):
        original = _unit(1, 0)
        transformed = _unit(1, 1)
        enroll = _unit(0, 1)
        embedder = StubEmbedder([original] + [transformed] * len(TRANSFORM_ORDER))

        reading = probe_crop(_crop(), enroll, embedder)
        first = reading.readings[0]

        self.assertAlmostEqual(first.cos_orig_enroll, 0.0)
        self.assertAlmostEqual(first.cos_transformed_enroll, np.sqrt(0.5))
        self.assertAlmostEqual(first.cos_orig_transformed, np.sqrt(0.5))

    def test_cos_orig_enroll_is_constant_across_transforms(self):
        vectors = [_unit(1, 0)] + [_unit(i + 1, 1) for i in range(len(TRANSFORM_ORDER))]
        embedder = StubEmbedder(vectors)

        reading = probe_crop(_crop(), _unit(0, 1), embedder)
        values = {item.cos_orig_enroll for item in reading.readings}

        self.assertEqual(len(values), 1)

    def test_both_gate_measures_derive_from_stored_cosines(self):
        original = _unit(1, 0)
        transformed = _unit(1, 1)
        enroll = _unit(0, 1)
        embedder = StubEmbedder([original] + [transformed] * len(TRANSFORM_ORDER))

        first = probe_crop(_crop(), enroll, embedder).readings[0]

        self.assertAlmostEqual(self_consistency(first), 1.0 - np.sqrt(0.5))
        self.assertAlmostEqual(template_shift(first), np.sqrt(0.5))

    def test_embed_ms_is_recorded(self):
        embedder = StubEmbedder([_unit(1, 0)] * (1 + len(TRANSFORM_ORDER)))
        reading = probe_crop(_crop(), _unit(1, 0), embedder)

        self.assertGreaterEqual(reading.embed_ms, 0.0)

    def test_every_transform_changes_the_image(self):
        crop = _crop()
        for name in TRANSFORM_ORDER:
            with self.subTest(transform=name):
                squeezed = TRANSFORMS[name](crop)
                self.assertEqual(squeezed.size, crop.size)
                self.assertFalse(
                    np.array_equal(np.asarray(squeezed), np.asarray(crop)),
                    f"{name} left the image unchanged",
                )


class JpegHeadroomTest(unittest.TestCase):
    """세션마다 JPEG 변환이 작동할 여지가 있었는지 기록하기 위한 지표."""

    def test_already_compressed_input_has_near_zero_headroom(self):
        from src.verification.defenses.squeeze_probe import jpeg

        once = jpeg(75)(_crop())
        self.assertLess(jpeg_headroom(once, quality=75), jpeg_headroom(_crop(), quality=75))

    def test_headroom_is_non_negative(self):
        self.assertGreaterEqual(jpeg_headroom(_crop()), 0.0)


class TransformSetTest(unittest.TestCase):
    def test_order_has_no_duplicates(self):
        self.assertEqual(len(TRANSFORM_ORDER), len(set(TRANSFORM_ORDER)))

    def test_every_transform_declares_parameters(self):
        self.assertEqual(set(TRANSFORM_PARAMS), set(TRANSFORM_ORDER))

    def test_core_and_baseline_are_disjoint_subsets_of_the_order(self):
        core, baseline = set(CORE_TRANSFORMS), set(BASELINE_TRANSFORMS)

        self.assertTrue(core.issubset(TRANSFORM_ORDER))
        self.assertTrue(baseline.issubset(TRANSFORM_ORDER))
        self.assertEqual(core & baseline, set())

    def test_baseline_keeps_the_currently_shipped_face_auth_transforms(self):
        """새 세트와 배포된 설정을 같은 조건에서 비교하기 위한 기준선이다."""
        self.assertIn("jpeg_q75", BASELINE_TRANSFORMS)
        self.assertIn("bit5_round", BASELINE_TRANSFORMS)


if __name__ == "__main__":
    unittest.main()
