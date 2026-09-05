"""FaceNetEmbedder 배치 처리.

PERF-001 실측에서 face_auth 게이트가 연구 트랙 프로브보다 7배 느렸다. 게이트는 9장,
프로브는 15장을 임베딩하는데 적게 처리하는 쪽이 느리다. 병목은 연산이 아니라 forward
호출당 고정 비용이며, 원인은 embed가 이미지를 한 장씩 호출하는 구조다.

docs/experiments/PERF-001-detector-latency.md 5.2절 참조.

배치화가 임베딩 값을 바꾸면 지금까지 산출한 모든 임계값이 무효가 된다. 값 동일성이
이 변경에서 가장 중요한 성질이다.
"""

import unittest

import numpy as np
from PIL import Image

from src.face_auth.inference.verifier import FaceNetEmbedder

# 부동소수점 누적 차이 허용치. 실측 최대 차이는 아래 테스트가 출력한다.
ATOL = 1e-5


def _images(count, size=160):
    rng = np.random.default_rng(0)
    return [
        Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
        for _ in range(count)
    ]


class CallCountTest(unittest.TestCase):
    """배치화가 실제로 됐는지 본다. 값만 보면 한 장씩 돌아도 통과한다."""

    def _fake_model(self, counter):
        import torch

        class Fake:
            def __call__(self, batch):
                counter.append(batch.shape[0])
                # (n, 512) 결정적 출력. 배치 위치마다 다른 값을 준다.
                n = batch.shape[0]
                base = torch.arange(n, dtype=torch.float32).reshape(n, 1)
                return base + torch.ones(n, 512)

        return Fake()

    def _patched(self, counter):
        import src.verification.defenses.facenet_embed as fe

        original = fe.get_model
        model = self._fake_model(counter)
        fe.get_model = lambda device=None: (model, __import__("torch").device("cpu"))
        return fe, original

    def test_forward_is_called_once_for_many_images(self):
        counter = []
        module, original = self._patched(counter)
        try:
            FaceNetEmbedder().embed(_images(9))
        finally:
            module.get_model = original

        self.assertEqual(len(counter), 1, "forward가 여러 번 호출됐다")
        self.assertEqual(counter[0], 9, "배치 크기가 이미지 수와 다르다")

    def test_no_forward_for_an_empty_list(self):
        counter = []
        module, original = self._patched(counter)
        try:
            result = FaceNetEmbedder().embed([])
        finally:
            module.get_model = original

        self.assertEqual(result, [])
        self.assertEqual(counter, [], "빈 목록인데 forward를 호출했다")


class ContractTest(unittest.TestCase):
    """반환 계약. 값이 같아도 형태가 달라지면 하위 연산이 깨진다."""

    def setUp(self):
        self.result = FaceNetEmbedder().embed(_images(3))

    def test_returns_one_vector_per_image(self):
        self.assertEqual(len(self.result), 3)

    def test_each_vector_is_one_dimensional_512(self):
        for vector in self.result:
            self.assertEqual(vector.shape, (512,))

    def test_dtype_stays_float32(self):
        """값 동일성 테스트에 묻으면 float64로 바꿔도 값이 같아 통과한다."""
        for vector in self.result:
            self.assertEqual(vector.dtype, np.float32)

    def test_vectors_are_l2_normalised(self):
        for vector in self.result:
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=4)


class BatchEqualsSequentialTest(unittest.TestCase):
    """배치화 전후로 값이 같아야 한다. 다르면 모든 임계값이 무효가 된다."""

    def _sequential(self, images):
        from src.verification.defenses.facenet_embed import get_embedding

        return [get_embedding(image, None).numpy() for image in images]

    def test_batch_matches_sequential_within_tolerance(self):
        images = _images(9)
        batched = FaceNetEmbedder().embed(images)
        sequential = self._sequential(images)

        differences = [
            float(np.max(np.abs(left - right)))
            for left, right in zip(batched, sequential)
        ]
        worst = max(differences)
        print(f"\n  배치 vs 순차 최대 절대 차이 {worst:.3e}  (허용 {ATOL:.0e})")
        if worst > ATOL * 0.5:
            print(
                f"  경고: 허용치의 {worst / ATOL:.0%}다. 배치가 커지면 깨질 수 있다."
            )

        for index, difference in enumerate(differences):
            with self.subTest(image=index):
                self.assertLess(difference, ATOL)

    def test_single_image_matches_sequential(self):
        images = _images(1)
        self.assertLess(
            float(np.max(np.abs(FaceNetEmbedder().embed(images)[0] - self._sequential(images)[0]))),
            ATOL,
        )

    def test_order_is_preserved(self):
        """배치 안에서 순서가 섞이면 프레임과 임베딩이 어긋난다."""
        images = _images(5)
        batched = FaceNetEmbedder().embed(images)
        sequential = self._sequential(images)

        for index in range(len(images)):
            with self.subTest(index=index):
                self.assertLess(
                    float(np.max(np.abs(batched[index] - sequential[index]))), ATOL
                )


if __name__ == "__main__":
    unittest.main()
