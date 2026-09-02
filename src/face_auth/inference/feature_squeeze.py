from __future__ import annotations

from dataclasses import dataclass
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


class FeatureSqueezeInspector:
    """Runs transform-consistency checks as a secondary security signal."""

    def __init__(
        self,
        embedder,
        detector: TransformConsistencyDetector,
        config: FeatureSqueezeConfig | None = None,
        template_detector: TemplateShiftDetector | None = None,
    ) -> None:
        self.embedder = embedder
        self.detector = detector
        self.config = config or FeatureSqueezeConfig()
        self.template_detector = template_detector

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
            transformed = _transforms(crop, self.config)
            embeddings = self.embedder.embed(transformed)

            frame_results = [self.detector.evaluate(original, embeddings)]
            if self.template_detector is not None:
                frame_results.append(
                    self.template_detector.evaluate(original, embeddings)
                )
            for result in frame_results:
                per_gate.setdefault(result.gate, []).append(result)

        if not per_gate:
            return tuple(
                GateResult(name, GateStatus.NOT_EVALUATED) for name in self.gate_names
            )
        return tuple(_worst(results) for results in per_gate.values())


def _worst(results: list[GateResult]) -> GateResult:
    failures = [result for result in results if result.status is GateStatus.FAIL]
    errors = [result for result in results if result.status is GateStatus.ERROR]
    return max(failures or errors or results, key=lambda item: item.score or 0.0)


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
