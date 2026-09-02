from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from src.face_auth.domain import reason_codes
from src.face_auth.domain.types import FramePacket, GateResult, GateStatus
from src.face_auth.inference.active_liveness import ActiveLivenessGate, LivenessSample
from src.face_auth.inference.camera_motion import CameraMotionGate
from src.face_auth.inference.continuity import IdentityContinuityGate
from src.face_auth.inference.content_replay import ContentReplayGate
from src.face_auth.inference.pipeline import (
    BaselineEvidencePipeline,
    PipelineObservation,
)
from src.face_auth.inference.passive_pad import PassivePADGate


class PADScorer(Protocol):
    def score(self, crops: list[Image.Image]) -> list[float]:
        ...


class LivenessEstimator(Protocol):
    def estimate(self, frame_id: int, face) -> LivenessSample:
        ...


class AdversarialInspector(Protocol):
    gate_names: tuple[str, ...]

    def evaluate(
        self,
        crops: list[Image.Image],
        original_embeddings: list[np.ndarray],
    ) -> tuple[GateResult, ...]:
        ...


@dataclass(frozen=True)
class FullPipelineObservation:
    baseline: PipelineObservation
    gate_results: tuple[GateResult, ...]


class FullEvidencePipeline:
    """Combines baseline identity evidence with all FULL-profile gates."""

    def __init__(
        self,
        baseline: BaselineEvidencePipeline,
        pad_scorer: PADScorer,
        pad_gate: PassivePADGate,
        liveness_estimator: LivenessEstimator,
        liveness_gate: ActiveLivenessGate,
        continuity_gate: IdentityContinuityGate,
        *,
        camera_motion_gate: CameraMotionGate | None = None,
        content_replay_gate: ContentReplayGate | None = None,
        adversarial_inspector: AdversarialInspector | None = None,
    ) -> None:
        self.baseline = baseline
        self.pad_scorer = pad_scorer
        self.pad_gate = pad_gate
        self.liveness_estimator = liveness_estimator
        self.liveness_gate = liveness_gate
        self.continuity_gate = continuity_gate
        self.camera_motion_gate = camera_motion_gate or CameraMotionGate()
        self.content_replay_gate = content_replay_gate or ContentReplayGate()
        self.adversarial_inspector = adversarial_inspector

    def evaluate(
        self,
        frames: list[FramePacket],
        *,
        challenge_kind: str,
        challenge_start_frame_id: int,
    ) -> FullPipelineObservation:
        baseline = self.baseline.evaluate(frames)
        gate_results = list(baseline.gate_results)
        if not _baseline_ready(baseline):
            gate_results.extend(
                _not_evaluated_full_gates(
                    _adversarial_gate_names(self.adversarial_inspector)
                )
            )
            return FullPipelineObservation(baseline, tuple(gate_results))

        evidence = list(baseline.evidence)
        bboxes = {item.frame.frame_id: item.face.bbox for item in evidence}
        gate_results.append(self.camera_motion_gate.evaluate(frames, bboxes))
        gate_results.append(self.content_replay_gate.evaluate(frames))
        crops = [item.face.crop for item in evidence]
        embeddings = [np.asarray(item.embedding) for item in evidence]

        try:
            gate_results.append(self.pad_gate.evaluate(self.pad_scorer.score(crops)))
        except Exception:
            gate_results.append(_model_error("passive_pad"))

        try:
            samples = [
                self.liveness_estimator.estimate(item.frame.frame_id, item.face)
                for item in evidence
            ]
            gate_results.append(
                self.liveness_gate.evaluate(
                    challenge_kind,
                    samples,
                    challenge_start_frame_id=challenge_start_frame_id,
                )
            )
        except Exception:
            gate_results.append(_model_error("active_liveness"))

        try:
            gate_results.append(self.continuity_gate.evaluate(embeddings))
        except Exception:
            gate_results.append(_model_error("continuity"))

        if self.adversarial_inspector is not None:
            try:
                gate_results.extend(
                    self.adversarial_inspector.evaluate(crops, embeddings)
                )
            except Exception:
                gate_results.extend(
                    _model_error(name)
                    for name in _adversarial_gate_names(self.adversarial_inspector)
                )

        return FullPipelineObservation(baseline, tuple(gate_results))


def _baseline_ready(observation: PipelineObservation) -> bool:
    return bool(observation.evidence) and all(
        result.status is GateStatus.PASS for result in observation.gate_results
    )


def _adversarial_gate_names(inspector) -> tuple[str, ...]:
    """Inspector가 실제로 내보내는 게이트 이름. 없으면 빈 튜플."""
    if inspector is None:
        return ()
    return tuple(getattr(inspector, "gate_names", ("adversarial",)))


def _not_evaluated_full_gates(adversarial_gates: tuple[str, ...]) -> list[GateResult]:
    names = [
        "camera_motion",
        "content_replay",
        "passive_pad",
        "active_liveness",
        "continuity",
        *adversarial_gates,
    ]
    return [
        GateResult(
            name,
            GateStatus.NOT_EVALUATED,
            reason_codes=(reason_codes.INSUFFICIENT_VALID_FRAMES,),
        )
        for name in names
    ]


def _model_error(gate: str) -> GateResult:
    return GateResult(
        gate,
        GateStatus.ERROR,
        reason_codes=(reason_codes.MODEL_ERROR,),
    )
