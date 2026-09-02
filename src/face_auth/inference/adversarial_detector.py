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


@dataclass(frozen=True)
class TemplateShiftDetectorConfig:
    max_template_shift: float
    threshold_version: str
    min_transforms: int = 2


class TemplateShiftDetector:
    """Detects transforms that move a probe toward or away from the enrolled template.

    TransformConsistencyDetector와 다른 것을 잰다. 그 게이트는 입력에 조작 흔적이
    있는지를 등록 템플릿과 무관하게 보고, 이 게이트는 그 조작이 등록자로 위장하는
    방향인지를 본다. 두 측정량은 변환 순위를 다르게 매기므로 독립 게이트로 유지한다.
    근거는 docs/experiments/EXP-DET-001-camera-squeeze-probe.md 1절.
    """

    gate = "adversarial_template"

    def __init__(
        self, config: TemplateShiftDetectorConfig, template: np.ndarray
    ) -> None:
        self.config = config
        self.template = normalize_embedding(np.asarray(template))

    def evaluate(
        self,
        original_embedding: np.ndarray,
        transformed_embeddings: list[np.ndarray],
    ) -> GateResult:
        if len(transformed_embeddings) < self.config.min_transforms:
            return GateResult(
                self.gate,
                GateStatus.NOT_EVALUATED,
                reason_codes=(reason_codes.INSUFFICIENT_VALID_FRAMES,),
                threshold_version=self.config.threshold_version,
            )
        original = normalize_embedding(original_embedding)
        baseline = float(np.dot(original, self.template))
        shifts = [
            abs(baseline - float(np.dot(normalize_embedding(embedding), self.template)))
            for embedding in transformed_embeddings
        ]
        score = float(max(shifts))
        detected = score >= self.config.max_template_shift
        return GateResult(
            self.gate,
            GateStatus.FAIL if detected else GateStatus.PASS,
            score=score,
            threshold=self.config.max_template_shift,
            reason_codes=(reason_codes.ADVERSARIAL_SUSPECTED,) if detected else (),
            threshold_version=self.config.threshold_version,
        )
