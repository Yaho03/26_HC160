"""시나리오 이름과 실제 측정 대상의 일치.

배치화(621520a) 이후 forward_only_single_gate 가 meta 로는 calls:9, batch_size:1 을
선언하면서 실제로는 배치 1회를 쟀다. 이름이 아니라 불일치 자체를 잡는 장치가 필요하다.

meta["calls"] 는 그 시나리오가 forward 를 몇 번 부르는지의 선언이다. 스텁 모델로
실제 호출 수를 세어 대조한다.
"""

import unittest

import numpy as np
from PIL import Image

from src.verification.defenses.latency_bench import forward_scenarios


class CountingModel:
    """forward 호출마다 배치 크기를 기록한다."""

    def __init__(self):
        self.batches = []

    def __call__(self, batch):
        import torch

        self.batches.append(int(batch.shape[0]))
        n = batch.shape[0]
        return torch.ones(n, 512)

    @property
    def calls(self):
        return len(self.batches)


def _crop(size=160):
    pixels = np.random.default_rng(0).integers(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


class ForwardScenarioContractTest(unittest.TestCase):
    GATE_IMAGES = 9

    def setUp(self):
        import src.verification.defenses.facenet_embed as fe
        import torch

        self.module = fe
        self.original = fe.get_model
        self.model = CountingModel()
        fe.get_model = lambda device=None: (self.model, torch.device("cpu"))

    def tearDown(self):
        self.module.get_model = self.original

    def _scenarios(self):
        from src.face_auth.inference.verifier import FaceNetEmbedder
        from src.verification.defenses.facenet_embed import FaceNetBatchEmbedder

        return forward_scenarios(
            crop=_crop(),
            gate_images=self.GATE_IMAGES,
            frame_embedder=FaceNetEmbedder(device="cpu"),
            batch_embedder=FaceNetBatchEmbedder(device="cpu"),
        )

    def test_every_scenario_declares_calls_and_batch_size(self):
        for name, (_, meta) in self._scenarios().items():
            with self.subTest(scenario=name):
                self.assertIn("calls", meta)
                self.assertIn("batch_size", meta)

    def test_declared_call_count_matches_the_actual_forward_count(self):
        """이름이 아니라 선언과 실제의 불일치를 잡는다."""
        for name, (step, meta) in self._scenarios().items():
            with self.subTest(scenario=name):
                self.model.batches.clear()
                step()
                self.assertEqual(
                    self.model.calls, meta["calls"],
                    f"{name}: meta는 forward {meta['calls']}회를 선언했으나 "
                    f"실제 {self.model.calls}회 호출됐다",
                )

    def test_declared_batch_size_matches_the_actual_batch_size(self):
        for name, (step, meta) in self._scenarios().items():
            with self.subTest(scenario=name):
                self.model.batches.clear()
                step()
                self.assertEqual(
                    set(self.model.batches), {meta["batch_size"]},
                    f"{name}: meta는 배치 {meta['batch_size']}를 선언했으나 "
                    f"실제 {self.model.batches}였다",
                )

    def test_loop_and_batch_scenarios_both_exist(self):
        """두 경로를 나눠 재야 호출 오버헤드가 드러난다."""
        names = set(self._scenarios())
        self.assertIn("forward_only_loop_gate", names)
        self.assertIn("forward_only_batch_gate", names)

    def test_the_misleading_name_is_gone(self):
        """621520a에서 측정 대상이 바뀐 이름은 남기지 않는다."""
        self.assertNotIn("forward_only_single_gate", self._scenarios())

    def test_loop_scenario_calls_forward_once_per_image(self):
        scenarios = self._scenarios()
        self.model.batches.clear()
        scenarios["forward_only_loop_gate"][0]()

        self.assertEqual(self.model.calls, self.GATE_IMAGES)
        self.assertEqual(set(self.model.batches), {1})

    def test_batch_scenario_calls_forward_once(self):
        scenarios = self._scenarios()
        self.model.batches.clear()
        scenarios["forward_only_batch_gate"][0]()

        self.assertEqual(self.model.calls, 1)
        self.assertEqual(self.model.batches, [self.GATE_IMAGES])


if __name__ == "__main__":
    unittest.main()
