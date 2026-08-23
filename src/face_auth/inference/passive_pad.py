from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.face_auth.domain import reason_codes
from src.face_auth.domain.types import GateResult, GateStatus


@dataclass(frozen=True)
class PassivePADConfig:
    live_threshold: float
    model_version: str
    threshold_version: str
    min_frames: int = 5


class PassivePADGate:
    """Aggregates calibrated live probabilities over a valid frame window."""

    def __init__(self, config: PassivePADConfig) -> None:
        self.config = config

    def evaluate(self, live_scores: list[float]) -> GateResult:
        if len(live_scores) < self.config.min_frames:
            return GateResult(
                "passive_pad",
                GateStatus.FAIL,
                reason_codes=(reason_codes.INSUFFICIENT_VALID_FRAMES,),
                model_version=self.config.model_version,
                threshold_version=self.config.threshold_version,
            )
        scores = np.asarray(live_scores, dtype=np.float32)
        if not np.isfinite(scores).all():
            return GateResult(
                "passive_pad",
                GateStatus.ERROR,
                reason_codes=(reason_codes.MODEL_ERROR,),
                model_version=self.config.model_version,
                threshold_version=self.config.threshold_version,
            )
        score = float(np.median(scores))
        passed = score >= self.config.live_threshold
        return GateResult(
            "passive_pad",
            GateStatus.PASS if passed else GateStatus.FAIL,
            score=score,
            threshold=self.config.live_threshold,
            reason_codes=() if passed else (reason_codes.SPOOF_SUSPECTED,),
            model_version=self.config.model_version,
            threshold_version=self.config.threshold_version,
        )
