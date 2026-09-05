"""Detector latency 계측의 순수 부분.

여기서 검증하는 것은 시간이 얼마나 걸리는지가 아니라 **집계와 계측 규칙**이다.
실제 소요 시간은 기기에 따라 달라지므로 단위 test의 통과 조건이 될 수 없다.
대신 다음을 고정한다.

- 백분위 정의 (nearest-rank). 정의를 적지 않은 p95는 재현되지 않는다.
- 워밍업 표본이 결과에서 빠진다. 모델 로딩과 커널 컴파일이 첫 호출에 섞인다.
- 표본 수가 결과에 남는다. 09_EVALUATION_METRICS.md 6절의 분모 명시 원칙이다.
- 단계 분해가 실제 호출을 반영한다. 배치 1회인지 프레임마다 1회인지가 병목을 가른다.
"""

import time
import unittest

import numpy as np
from PIL import Image

from src.face_auth.inference.adversarial_detector import (
    AdversarialDetectorConfig,
    TemplateShiftDetector,
    TemplateShiftDetectorConfig,
    TransformConsistencyDetector,
)
from src.face_auth.inference.feature_squeeze import FeatureSqueezeConfig
from src.verification.defenses.latency_bench import (
    EmptySampleError,
    callable_step,
    ForwardClock,
    TimedBatchEmbedder,
    TimedFrameEmbedder,
    fixed_transform_step,
    gate_step,
    gate_transform_step,
    percentile_ms,
    probe_step,
    randomized_transform_step,
    run_repeats,
    summarize,
    summarize_stages,
    synthetic_crop,
)
from src.verification.defenses.squeeze_probe import TRANSFORM_ORDER


def _unit(*components):
    vector = np.asarray(components, dtype=np.float64)
    return vector / np.linalg.norm(vector)


class StubBatchEmbedder:
    """호출당 이미지 수를 기록한다. 배치가 1회인지 확인하기 위해서다."""

    def __init__(self, dim=8, sleep_s=0.0):
        self.dim = dim
        self.sleep_s = sleep_s
        self.batch_sizes = []

    def embed_batch(self, images):
        self.batch_sizes.append(len(images))
        if self.sleep_s:
            time.sleep(self.sleep_s)
        matrix = np.zeros((len(images), self.dim), dtype=np.float64)
        matrix[:, 0] = 1.0
        return matrix


class StubFrameEmbedder:
    """face_auth 임베더 계약. `embed(images) -> list[np.ndarray]`."""

    def __init__(self, dim=8, sleep_s=0.0):
        self.dim = dim
        self.sleep_s = sleep_s
        self.batch_sizes = []

    def embed(self, images):
        self.batch_sizes.append(len(images))
        if self.sleep_s:
            time.sleep(self.sleep_s)
        vector = np.zeros(self.dim, dtype=np.float64)
        vector[0] = 1.0
        return [vector.copy() for _ in images]


class PercentileTest(unittest.TestCase):
    def test_nearest_rank_p95_of_one_hundred_is_the_ninety_fifth_value(self):
        samples = [float(value) for value in range(1, 101)]
        self.assertEqual(percentile_ms(samples, 95), 95.0)

    def test_p50_of_one_hundred_is_the_fiftieth_value(self):
        samples = [float(value) for value in range(1, 101)]
        self.assertEqual(percentile_ms(samples, 50), 50.0)

    def test_unsorted_input_gives_the_same_answer(self):
        samples = [5.0, 1.0, 4.0, 2.0, 3.0]
        self.assertEqual(percentile_ms(samples, 100), 5.0)
        self.assertEqual(percentile_ms(samples, 20), 1.0)

    def test_single_sample_is_its_own_percentile(self):
        self.assertEqual(percentile_ms([7.5], 95), 7.5)

    def test_empty_samples_raise_instead_of_returning_zero(self):
        """표본이 없으면 0이 아니라 오류다. 0은 '빠르다'로 읽힌다."""
        with self.assertRaises(EmptySampleError):
            percentile_ms([], 95)


class SummaryTest(unittest.TestCase):
    def test_summary_keeps_the_sample_count(self):
        summary = summarize("probe", [1.0, 2.0, 3.0, 4.0])

        self.assertEqual(summary.n, 4)
        self.assertEqual(summary.label, "probe")

    def test_summary_reports_p50_p95_mean_and_range(self):
        summary = summarize("probe", [float(value) for value in range(1, 21)])

        self.assertEqual(summary.p50_ms, 10.0)
        self.assertEqual(summary.p95_ms, 19.0)
        self.assertAlmostEqual(summary.mean_ms, 10.5)
        self.assertEqual(summary.min_ms, 1.0)
        self.assertEqual(summary.max_ms, 20.0)

    def test_as_dict_exposes_n_so_reports_cannot_drop_the_denominator(self):
        payload = summarize("probe", [1.0, 2.0]).as_dict()

        self.assertEqual(payload["n"], 2)
        self.assertIn("p95_ms", payload)

    def test_empty_samples_raise(self):
        with self.assertRaises(EmptySampleError):
            summarize("probe", [])


class RunRepeatsTest(unittest.TestCase):
    def test_warmup_samples_are_excluded_from_the_result(self):
        calls = []

        def step():
            calls.append(len(calls))
            return {"total": float(len(calls))}

        records = run_repeats(step, repeats=5, warmup=3)

        self.assertEqual(len(calls), 8)
        self.assertEqual(len(records), 5)
        self.assertEqual([record["total"] for record in records], [4.0, 5.0, 6.0, 7.0, 8.0])

    def test_zero_warmup_is_allowed(self):
        records = run_repeats(lambda: {"total": 1.0}, repeats=2, warmup=0)

        self.assertEqual(len(records), 2)

    def test_non_positive_repeats_is_rejected(self):
        with self.assertRaises(ValueError):
            run_repeats(lambda: {"total": 1.0}, repeats=0, warmup=1)

    def test_negative_warmup_is_rejected(self):
        with self.assertRaises(ValueError):
            run_repeats(lambda: {"total": 1.0}, repeats=1, warmup=-1)


class SummarizeStagesTest(unittest.TestCase):
    def test_one_summary_per_stage(self):
        records = [
            {"total": 10.0, "forward": 7.0, "other": 3.0},
            {"total": 12.0, "forward": 9.0, "other": 3.0},
        ]

        summaries = summarize_stages("gate", records)

        self.assertEqual(set(summaries), {"total", "forward", "other"})
        self.assertEqual(summaries["forward"].n, 2)
        self.assertEqual(summaries["total"].label, "gate.total")

    def test_missing_stage_in_one_record_is_rejected(self):
        records = [{"total": 1.0, "forward": 1.0}, {"total": 1.0}]

        with self.assertRaises(ValueError):
            summarize_stages("gate", records)


class ForwardClockTest(unittest.TestCase):
    def test_clock_accumulates_across_calls_within_one_iteration(self):
        clock = ForwardClock()
        clock.add(1.5)
        clock.add(2.5)

        self.assertAlmostEqual(clock.elapsed_ms, 4.0)

        clock.reset()
        self.assertEqual(clock.elapsed_ms, 0.0)

    def test_timed_batch_embedder_passes_values_through(self):
        inner = StubBatchEmbedder()
        clock = ForwardClock()
        timed = TimedBatchEmbedder(inner, clock)

        result = timed.embed_batch([synthetic_crop(32)] * 3)

        self.assertEqual(result.shape, (3, inner.dim))
        self.assertEqual(inner.batch_sizes, [3])
        self.assertGreaterEqual(clock.elapsed_ms, 0.0)

    def test_timed_frame_embedder_passes_values_through(self):
        inner = StubFrameEmbedder()
        clock = ForwardClock()
        timed = TimedFrameEmbedder(inner, clock)

        result = timed.embed([synthetic_crop(32)] * 2)

        self.assertEqual(len(result), 2)
        self.assertEqual(inner.batch_sizes, [2])

    def test_measured_forward_time_tracks_the_inner_call(self):
        inner = StubBatchEmbedder(sleep_s=0.02)
        clock = ForwardClock()

        TimedBatchEmbedder(inner, clock).embed_batch([synthetic_crop(32)])

        self.assertGreaterEqual(clock.elapsed_ms, 15.0)


class ProbeScenarioTest(unittest.TestCase):
    def test_research_probe_uses_a_single_batch_of_original_plus_transforms(self):
        embedder = StubBatchEmbedder()
        step = probe_step(synthetic_crop(160), _unit(1, 0, 0, 0, 0, 0, 0, 0), embedder)

        step()

        self.assertEqual(embedder.batch_sizes, [1 + len(TRANSFORM_ORDER)])

    def test_probe_step_reports_total_forward_and_other(self):
        embedder = StubBatchEmbedder()
        step = probe_step(synthetic_crop(160), _unit(1, 0, 0, 0, 0, 0, 0, 0), embedder)

        record = step()

        self.assertEqual(set(record), {"total", "forward", "other"})
        self.assertAlmostEqual(record["total"], record["forward"] + record["other"], places=6)

    def test_forward_time_is_a_share_of_the_total(self):
        embedder = StubBatchEmbedder(sleep_s=0.02)
        step = probe_step(synthetic_crop(160), _unit(1, 0, 0, 0, 0, 0, 0, 0), embedder)

        record = step()

        self.assertGreaterEqual(record["forward"], 15.0)
        self.assertGreaterEqual(record["total"], record["forward"])


class GateScenarioTest(unittest.TestCase):
    def _detector(self):
        return TransformConsistencyDetector(
            AdversarialDetectorConfig(
                max_cosine_distance=0.5, threshold_version="latency-bench"
            )
        )

    def _originals(self, count):
        return [_unit(1, 0, 0, 0, 0, 0, 0, 0) for _ in range(count)]

    def test_gate_embeds_three_transforms_once_per_evaluated_frame(self):
        embedder = StubFrameEmbedder()
        config = FeatureSqueezeConfig(max_frames=3)
        step = gate_step(
            embedder,
            self._detector(),
            [synthetic_crop(160) for _ in range(3)],
            self._originals(3),
            config,
        )

        step()

        self.assertEqual(embedder.batch_sizes, [3, 3, 3])

    def test_gate_only_evaluates_the_last_max_frames(self):
        """윈도보다 많은 프레임을 줘도 비용은 윈도 크기로 묶인다."""
        embedder = StubFrameEmbedder()
        config = FeatureSqueezeConfig(max_frames=2)
        step = gate_step(
            embedder,
            self._detector(),
            [synthetic_crop(160) for _ in range(5)],
            self._originals(5),
            config,
        )

        step()

        self.assertEqual(embedder.batch_sizes, [3, 3])

    def test_template_gate_shares_transforms_and_embeddings(self):
        """게이트를 하나 더 붙여도 forward 수가 늘면 안 된다."""
        embedder = StubFrameEmbedder()
        template_detector = TemplateShiftDetector(
            TemplateShiftDetectorConfig(
                max_template_shift=0.5, threshold_version="latency-bench"
            ),
            _unit(1, 0, 0, 0, 0, 0, 0, 0),
        )
        step = gate_step(
            embedder,
            self._detector(),
            [synthetic_crop(160) for _ in range(3)],
            self._originals(3),
            FeatureSqueezeConfig(max_frames=3),
            template_detector=template_detector,
        )

        step()

        self.assertEqual(embedder.batch_sizes, [3, 3, 3])

    def test_gate_step_reports_total_forward_and_other(self):
        embedder = StubFrameEmbedder()
        step = gate_step(
            embedder,
            self._detector(),
            [synthetic_crop(160) for _ in range(3)],
            self._originals(3),
            FeatureSqueezeConfig(max_frames=3),
        )

        record = step()

        self.assertEqual(set(record), {"total", "forward", "other"})
        self.assertAlmostEqual(record["total"], record["forward"] + record["other"], places=6)


class TransformOnlyStepTest(unittest.TestCase):
    def test_fixed_step_applies_every_named_transform(self):
        applied = []
        crop = synthetic_crop(160)
        step = fixed_transform_step(crop, ["blur0.8", "median3"], sink=applied.append)

        record = step()

        self.assertEqual(len(applied), 2)
        self.assertEqual(set(record), {"total"})

    def test_unknown_fixed_transform_is_rejected(self):
        with self.assertRaises(KeyError):
            fixed_transform_step(synthetic_crop(32), ["does_not_exist"])

    def test_gate_transform_step_produces_three_images(self):
        applied = []
        step = gate_transform_step(
            synthetic_crop(160), FeatureSqueezeConfig(), sink=applied.append
        )

        step()

        self.assertEqual(len(applied), 1)
        self.assertEqual(len(applied[0]), 3)

    def test_randomized_step_draws_new_parameters_each_iteration(self):
        """파라미터가 매번 달라져야 캐시 효과가 사라진다는 전제를 코드로 고정한다."""
        drawn = []
        rng = np.random.default_rng(0)
        step = randomized_transform_step(
            synthetic_crop(160), ["blur", "jpeg", "median"], rng, sink=drawn.append
        )

        step()
        step()

        self.assertEqual(len(drawn), 6)
        blur_params = [spec.params["radius"] for spec in drawn if spec.family == "blur"]
        self.assertNotEqual(blur_params[0], blur_params[1])

    def test_randomized_step_is_reproducible_from_a_seed(self):
        def radii(seed):
            drawn = []
            step = randomized_transform_step(
                synthetic_crop(64),
                ["blur"],
                np.random.default_rng(seed),
                sink=drawn.append,
            )
            step()
            step()
            return [spec.params["radius"] for spec in drawn]

        self.assertEqual(radii(7), radii(7))

    def test_randomized_and_fixed_steps_apply_the_same_transform_count(self):
        """비교 가능한 조건인지 확인한다. 개수가 다르면 차이가 랜덤화 비용이 아니다."""
        fixed_applied = []
        random_applied = []
        crop = synthetic_crop(160)

        fixed_transform_step(
            crop, ["blur0.8", "jpeg_q75", "median3"], sink=fixed_applied.append
        )()
        randomized_transform_step(
            crop,
            ["blur", "jpeg", "median"],
            np.random.default_rng(0),
            sink=random_applied.append,
        )()

        self.assertEqual(len(fixed_applied), len(random_applied))


class CallableStepTest(unittest.TestCase):
    """forward를 변환에서 떼어 내 단독으로 재기 위한 최소 계측."""

    def test_step_invokes_the_callable_once_per_iteration(self):
        calls = []
        step = callable_step(lambda: calls.append(1))

        step()
        step()

        self.assertEqual(len(calls), 2)

    def test_step_reports_only_total(self):
        record = callable_step(lambda: None)()

        self.assertEqual(set(record), {"total"})

    def test_step_measures_the_call_duration(self):
        record = callable_step(lambda: time.sleep(0.02))()

        self.assertGreaterEqual(record["total"], 15.0)


class SyntheticCropTest(unittest.TestCase):
    def test_crop_is_square_rgb_at_the_requested_size(self):
        crop = synthetic_crop(160)

        self.assertIsInstance(crop, Image.Image)
        self.assertEqual(crop.size, (160, 160))
        self.assertEqual(crop.mode, "RGB")

    def test_crop_is_deterministic_for_a_seed(self):
        left = np.asarray(synthetic_crop(64, seed=3))
        right = np.asarray(synthetic_crop(64, seed=3))

        self.assertTrue(np.array_equal(left, right))

    def test_crop_is_not_uniform_noise(self):
        """균일 잡음은 JPEG가 거의 압축하지 못해 실제 얼굴보다 느리게 나온다."""
        crop = synthetic_crop(160, seed=0)
        array = np.asarray(crop, dtype=np.float64)
        horizontal_gradient = np.abs(np.diff(array, axis=1)).mean()

        self.assertLess(horizontal_gradient, 20.0)


if __name__ == "__main__":
    unittest.main()
