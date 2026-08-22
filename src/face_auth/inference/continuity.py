from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.face_auth.domain import reason_codes
from src.face_auth.domain.types import GateResult, GateStatus
from src.face_auth.inference.verifier import normalize_embedding


@dataclass(frozen=True)
class ContinuityConfig:
    min_anchor_similarity: float = 0.65
    window_size: int = 5
    failures_required: int = 3
    threshold_version: str = "continuity-3-of-5-v1"


class IdentityContinuityGate:
    def __init__(
        self, anchor_embedding: np.ndarray, config: ContinuityConfig | None = None
    ) -> None:
        self.anchor = normalize_embedding(anchor_embedding)
        self.config = config or ContinuityConfig()
        if not 1 <= self.config.failures_required <= self.config.window_size:
            raise ValueError("failures_required must be within the continuity window")

    def evaluate(self, embeddings: list[np.ndarray]) -> GateResult:
        if len(embeddings) < self.config.window_size:
            return GateResult(
                "continuity",
                GateStatus.FAIL,
                reason_codes=(reason_codes.INSUFFICIENT_VALID_FRAMES,),
                threshold=self.config.min_anchor_similarity,
                threshold_version=self.config.threshold_version,
            )
        window = embeddings[-self.config.window_size :]
        similarities = [
            float(np.dot(normalize_embedding(embedding), self.anchor))
            for embedding in window
        ]
        failures = sum(
            score < self.config.min_anchor_similarity for score in similarities
        )
        passed = failures < self.config.failures_required
        return GateResult(
            "continuity",
            GateStatus.PASS if passed else GateStatus.FAIL,
            score=float(np.median(similarities)),
            threshold=self.config.min_anchor_similarity,
            reason_codes=() if passed else (reason_codes.IDENTITY_SWITCH,),
            threshold_version=self.config.threshold_version,
        )
