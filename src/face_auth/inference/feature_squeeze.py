from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter

from src.face_auth.domain.types import GateResult, GateStatus
from src.face_auth.inference.adversarial_detector import TransformConsistencyDetector


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
    ) -> None:
        self.embedder = embedder
        self.detector = detector
        self.config = config or FeatureSqueezeConfig()

    def evaluate(
        self,
        crops: list[Image.Image],
        original_embeddings: list[np.ndarray],
    ) -> GateResult:
        results: list[GateResult] = []
        for crop, original in list(zip(crops, original_embeddings))[
            -self.config.max_frames :
        ]:
            transformed = _transforms(crop, self.config)
            embeddings = self.embedder.embed(transformed)
            results.append(self.detector.evaluate(original, embeddings))

        if not results:
            return GateResult("adversarial", GateStatus.NOT_EVALUATED)
        failures = [result for result in results if result.status is GateStatus.FAIL]
        errors = [result for result in results if result.status is GateStatus.ERROR]
        selected = max(
            failures or errors or results, key=lambda item: item.score or 0.0
        )
        return selected


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
