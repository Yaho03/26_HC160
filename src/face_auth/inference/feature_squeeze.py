"""
Squeezing 기반 secondary inspection.

변환 파라미터를 고정하면 공격자가 방어를 알 때 그대로 재현해 우회할 수 있다. 연구
트랙 실측에서 고정 변환의 BPDA 탐지율이 0%였고, 파라미터를 매번 무작위로 뽑으니
84%로 회복됐다. docs/experiments/EXP-DET-001-camera-squeeze-probe.md 12절과 16절을
참조한다.

랜덤화는 기본값이 꺼짐이다. face_auth 경로에서의 탐지율은 아직 측정하지 않았다.
연구 트랙의 결과는 verification 경로에서 얻은 것이며 그대로 이 경로에 적용되지 않는다.

변환 계열은 배포된 3종(jpeg, bit, blur)을 유지한다. 연구 트랙에서 bit 계열이 약한
것으로 나타났으나 face_auth 경로에서 비교한 적이 없다. 계열 교체는 랜덤화와 분리해
별도로 판단한다. 둘을 한 번에 바꾸면 탐지율 변화의 원인을 가릴 수 없다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter

from src.face_auth.domain.types import GateResult, GateStatus
from src.face_auth.inference.adversarial_detector import (
    TemplateShiftDetector,
    TransformConsistencyDetector,
)


@dataclass(frozen=True)
class FeatureSqueezeConfig:
    jpeg_quality: int = 75
    bit_depth: int = 5
    blur_radius: float = 0.8
    max_frames: int = 3

    # 랜덤화. 검증되지 않은 변경이므로 기본값은 꺼짐이다.
    randomized: bool = False
    # 계열과 범위 프리셋을 명시적으로 둔다. 연구 트랙 모듈의 기본값에 기대면 그쪽에서
    # 범위를 바꿀 때 인증 동작이 조용히 바뀐다. ADR-003은 임계값을 preprocessing에 묶인
    # 버전 artifact로 다루라고 하며 변환 범위도 preprocessing의 일부다.
    families: tuple[str, ...] = ("jpeg", "bit", "blur")
    range_preset: str = "narrow"


class FeatureSqueezeInspector:
    """Runs transform-consistency checks as a secondary security signal."""

    def __init__(
        self,
        embedder,
        detector: TransformConsistencyDetector,
        config: FeatureSqueezeConfig | None = None,
        template_detector: TemplateShiftDetector | None = None,
        rng=None,
    ) -> None:
        self.embedder = embedder
        self.detector = detector
        self.config = config or FeatureSqueezeConfig()
        self.template_detector = template_detector
        # 주입 가능하게 두어 테스트에서 재현할 수 있게 한다.
        self.rng = rng

    def _rng(self):
        if not self.config.randomized:
            return None
        if self.rng is None:
            self.rng = np.random.default_rng()
        return self.rng

    @property
    def gate_names(self) -> tuple[str, ...]:
        names = ["adversarial"]
        if self.template_detector is not None:
            names.append(self.template_detector.gate)
        return tuple(names)

    def evaluate(
        self,
        crops: list[Image.Image],
        original_embeddings: list[np.ndarray],
    ) -> tuple[GateResult, ...]:
        """게이트마다 가장 나쁜 프레임의 결과를 하나씩 돌려준다.

        변환과 임베딩은 프레임당 한 번만 계산해 두 게이트가 공유한다. 게이트별로
        따로 돌리면 FaceNet forward가 두 배가 된다.
        """
        per_gate: dict[str, list[GateResult]] = {}
        for crop, original in list(zip(crops, original_embeddings))[
            -self.config.max_frames :
        ]:
            # 프레임마다 새로 뽑는다. 한 번 뽑아 재사용하면 공격자가 한 번만 맞히면 된다.
            transformed, evidence = _transforms_with_evidence(
                crop, self.config, self._rng()
            )
            embeddings = self.embedder.embed(transformed)

            frame_results = [self.detector.evaluate(original, embeddings)]
            if self.template_detector is not None:
                frame_results.append(
                    self.template_detector.evaluate(original, embeddings)
                )
            # 두 게이트가 같은 변환본을 썼으므로 같은 evidence를 갖는다.
            for result in frame_results:
                per_gate.setdefault(result.gate, []).append(
                    replace(result, evidence=evidence) if evidence else result
                )

        if not per_gate:
            return tuple(
                GateResult(name, GateStatus.NOT_EVALUATED) for name in self.gate_names
            )
        return tuple(_worst(results) for results in per_gate.values())


def _worst(results: list[GateResult]) -> GateResult:
    failures = [result for result in results if result.status is GateStatus.FAIL]
    errors = [result for result in results if result.status is GateStatus.ERROR]
    return max(failures or errors or results, key=lambda item: item.score or 0.0)


def _transforms_with_evidence(image, config, rng):
    """
    변환본과 그 파라미터를 함께 돌려준다.

    고정 변환은 config에 이미 적혀 있으므로 evidence를 비워 둔다. 랜덤화한 경우에만
    재현에 필요한 파라미터를 남긴다.
    """
    if not config.randomized:
        return _transforms(image, config), ()

    from src.verification.defenses.randomized_squeeze import sample_transform

    images, evidence = [], []
    for family in config.families:
        spec = sample_transform(family, rng, preset=config.range_preset)
        images.append(spec.apply(image))
        params = ",".join(f"{k}={v:.6g}" for k, v in spec.params.items())
        evidence.append(f"{spec.family}:{params}")
    return images, tuple(evidence)


def _transforms(image: Image.Image, config: FeatureSqueezeConfig) -> list[Image.Image]:
    jpeg_buffer = BytesIO()
    image.convert("RGB").save(jpeg_buffer, format="JPEG", quality=config.jpeg_quality)
    jpeg_buffer.seek(0)
    jpeg = Image.open(jpeg_buffer).convert("RGB").copy()

    levels = float((1 << config.bit_depth) - 1)
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    reduced = Image.fromarray(
        np.clip(np.rint(array * levels) / levels * 255.0, 0, 255).astype(np.uint8)
    )
    blurred = image.convert("RGB").filter(ImageFilter.GaussianBlur(config.blur_radius))
    return [jpeg, reduced, blurred]
