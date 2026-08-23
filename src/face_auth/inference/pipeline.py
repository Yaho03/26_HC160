from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from src.face_auth.domain import reason_codes
from src.face_auth.domain.types import FramePacket, GateResult, GateStatus
from src.face_auth.inference.face_detector import DetectedFace, face_count_gate
from src.face_auth.inference.frame_integrity import FrameIntegrityGate
from src.face_auth.inference.quality import QualityGate
from src.face_auth.inference.verifier import MultiFrameVerifier


class FaceDetector(Protocol):
    def detect(self, frame_bgr) -> list[DetectedFace]:
        ...


class Embedder(Protocol):
    def embed(self, images: list[Image.Image]) -> list:
        ...


@dataclass(frozen=True)
class ValidFaceEvidence:
    frame: FramePacket
    face: DetectedFace
    embedding: object


@dataclass(frozen=True)
class PipelineObservation:
    gate_results: tuple[GateResult, ...]
    total_frames: int
    valid_face_frames: int
    evidence: tuple[ValidFaceEvidence, ...] = ()


class BaselineEvidencePipeline:
    """Recorded-video/webcam baseline before PAD and continuity are enabled."""

    def __init__(
        self,
        detector: FaceDetector,
        embedder: Embedder,
        verifier: MultiFrameVerifier,
        *,
        quality: QualityGate | None = None,
        min_valid_frames: int = 3,
    ) -> None:
        self.detector = detector
        self.embedder = embedder
        self.verifier = verifier
        self.quality = quality or QualityGate()
        self.min_valid_frames = min_valid_frames

    def evaluate(self, frames: list[FramePacket]) -> PipelineObservation:
        integrity = FrameIntegrityGate()
        frame_integrity_result = GateResult("frame_integrity", GateStatus.PASS)
        valid_pairs: list[tuple[FramePacket, DetectedFace]] = []
        quality_failures: list[GateResult] = []
        saw_single_face = False
        multiple_faces = False

        for frame in frames:
            frame_result = integrity.evaluate(frame)
            if frame_result.status is GateStatus.FAIL:
                frame_integrity_result = frame_result
                break
            try:
                faces = self.detector.detect(frame.image_bgr)
            except Exception:
                return PipelineObservation(
                    gate_results=(
                        frame_integrity_result,
                        GateResult(
                            "single_face",
                            GateStatus.ERROR,
                            reason_codes=(reason_codes.MODEL_ERROR,),
                        ),
                        GateResult("quality", GateStatus.NOT_EVALUATED),
                        GateResult("identity", GateStatus.NOT_EVALUATED),
                    ),
                    total_frames=len(frames),
                    valid_face_frames=0,
                )

            count_result = face_count_gate(faces)
            if count_result.reason_codes == (reason_codes.MULTIPLE_FACES,):
                multiple_faces = True
                break
            if count_result.status is not GateStatus.PASS:
                continue

            saw_single_face = True
            face = faces[0]
            quality = self.quality.evaluate(frame.image_bgr, face.bbox).result
            if quality.status is GateStatus.PASS:
                valid_pairs.append((frame, face))
            else:
                quality_failures.append(quality)

        single_face_result = self._single_face_result(saw_single_face, multiple_faces)
        quality_result = self._quality_result(valid_pairs, quality_failures)
        evidence: tuple[ValidFaceEvidence, ...] = ()

        if (
            frame_integrity_result.status is GateStatus.PASS
            and single_face_result.status is GateStatus.PASS
            and quality_result.status is GateStatus.PASS
        ):
            try:
                embeddings = self.embedder.embed([face.crop for _, face in valid_pairs])
                identity_result = self.verifier.evaluate(embeddings)
                evidence = tuple(
                    ValidFaceEvidence(frame=frame, face=face, embedding=embedding)
                    for (frame, face), embedding in zip(valid_pairs, embeddings)
                )
            except Exception:
                identity_result = GateResult(
                    "identity",
                    GateStatus.ERROR,
                    reason_codes=(reason_codes.MODEL_ERROR,),
                )
        else:
            identity_result = GateResult(
                "identity",
                GateStatus.NOT_EVALUATED,
                reason_codes=(reason_codes.INSUFFICIENT_VALID_FRAMES,),
            )

        return PipelineObservation(
            gate_results=(
                frame_integrity_result,
                quality_result,
                single_face_result,
                identity_result,
            ),
            total_frames=len(frames),
            valid_face_frames=len(valid_pairs),
            evidence=evidence,
        )

    def _quality_result(
        self,
        valid_pairs: list[tuple[FramePacket, DetectedFace]],
        failures: list[GateResult],
    ) -> GateResult:
        if len(valid_pairs) >= self.min_valid_frames:
            return GateResult(
                "quality",
                GateStatus.PASS,
                score=float(len(valid_pairs)),
                threshold=float(self.min_valid_frames),
                threshold_version="quality-window-v1",
            )
        reasons = tuple(
            dict.fromkeys(code for failure in failures for code in failure.reason_codes)
        )
        if not reasons:
            reasons = (reason_codes.INSUFFICIENT_VALID_FRAMES,)
        return GateResult(
            "quality",
            GateStatus.FAIL,
            score=float(len(valid_pairs)),
            threshold=float(self.min_valid_frames),
            reason_codes=reasons,
            threshold_version="quality-window-v1",
        )

    @staticmethod
    def _single_face_result(saw_single: bool, saw_multiple: bool) -> GateResult:
        if saw_multiple:
            return GateResult(
                "single_face",
                GateStatus.FAIL,
                reason_codes=(reason_codes.MULTIPLE_FACES,),
            )
        if not saw_single:
            return GateResult(
                "single_face",
                GateStatus.FAIL,
                reason_codes=(reason_codes.NO_FACE,),
            )
        return GateResult("single_face", GateStatus.PASS)
