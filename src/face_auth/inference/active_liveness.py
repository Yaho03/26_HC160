from __future__ import annotations

from dataclasses import dataclass

from src.face_auth.domain import reason_codes
from src.face_auth.domain.types import GateResult, GateStatus


@dataclass(frozen=True)
class LivenessSample:
    frame_id: int
    yaw_degrees: float
    eyes_closed: bool


@dataclass(frozen=True)
class ActiveLivenessConfig:
    head_turn_degrees: float = 15.0
    min_samples: int = 3
    threshold_version: str = "active-liveness-v1"


class ActiveLivenessGate:
    def __init__(self, config: ActiveLivenessConfig | None = None) -> None:
        self.config = config or ActiveLivenessConfig()

    def evaluate(
        self,
        challenge_kind: str,
        samples: list[LivenessSample],
        *,
        challenge_start_frame_id: int,
    ) -> GateResult:
        eligible = [
            sample for sample in samples if sample.frame_id > challenge_start_frame_id
        ]
        if len(eligible) < self.config.min_samples:
            return self._failed(reason_codes.INSUFFICIENT_VALID_FRAMES)

        if challenge_kind == "HEAD_LEFT":
            passed = (
                min(sample.yaw_degrees for sample in eligible)
                <= -self.config.head_turn_degrees
            )
        elif challenge_kind == "HEAD_RIGHT":
            passed = (
                max(sample.yaw_degrees for sample in eligible)
                >= self.config.head_turn_degrees
            )
        elif challenge_kind == "BLINK":
            passed = _contains_blink(eligible)
        else:
            return GateResult(
                "active_liveness",
                GateStatus.ERROR,
                reason_codes=(reason_codes.UNSUPPORTED_CHALLENGE,),
                threshold_version=self.config.threshold_version,
            )

        return GateResult(
            "active_liveness",
            GateStatus.PASS if passed else GateStatus.FAIL,
            threshold=self.config.head_turn_degrees
            if challenge_kind.startswith("HEAD_")
            else None,
            reason_codes=() if passed else (reason_codes.LIVENESS_FAILED,),
            threshold_version=self.config.threshold_version,
        )

    def _failed(self, reason: str) -> GateResult:
        return GateResult(
            "active_liveness",
            GateStatus.FAIL,
            reason_codes=(reason,),
            threshold_version=self.config.threshold_version,
        )


def _contains_blink(samples: list[LivenessSample]) -> bool:
    states = [sample.eyes_closed for sample in samples]
    return any(
        not states[index - 1] and states[index] and not states[index + 1]
        for index in range(1, len(states) - 1)
    )
