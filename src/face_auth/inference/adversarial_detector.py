from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.face_auth.domain import reason_codes
from src.face_auth.domain.types import GateResult, GateStatus
from src.face_auth.inference.verifier import normalize_embedding


@dataclass(frozen=True)
class AdversarialDetectorConfig:
    max_cosine_distance: float
    threshold_version: str
    min_transforms: int = 2


class TransformConsistencyDetector:
    """Detects instability between original and transformed face embeddings."""

    def __init__(self, config: AdversarialDetectorConfig) -> None:
        self.config = config

    def evaluate(
        self,
        original_embedding: np.ndarray,
        transformed_embeddings: list[np.ndarray],
    ) -> GateResult:
        if len(transformed_embeddings) < self.config.min_transforms:
            return GateResult(
                "adversarial",
                GateStatus.NOT_EVALUATED,
                reason_codes=(reason_codes.INSUFFICIENT_VALID_FRAMES,),
                threshold_version=self.config.threshold_version,
            )
        original = normalize_embedding(original_embedding)
        distances = [
            1.0 - float(np.dot(original, normalize_embedding(embedding)))
            for embedding in transformed_embeddings
        ]
        score = float(max(distances))
        detected = score >= self.config.max_cosine_distance
        return GateResult(
            "adversarial",
            GateStatus.FAIL if detected else GateStatus.PASS,
            score=score,
            threshold=self.config.max_cosine_distance,
            reason_codes=(reason_codes.ADVERSARIAL_SUSPECTED,) if detected else (),
            threshold_version=self.config.threshold_version,
        )
