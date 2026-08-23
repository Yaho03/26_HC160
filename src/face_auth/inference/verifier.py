from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.face_auth.domain import reason_codes
from src.face_auth.domain.types import GateResult, GateStatus


@dataclass(frozen=True)
class VerificationConfig:
    threshold: float
    threshold_version: str
    model_version: str
    min_probe_frames: int = 3
    min_enrollment_frames: int = 5


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Zero embedding cannot be normalized")
    return vector / norm


def build_template(
    embeddings: list[np.ndarray],
    *,
    min_frames: int = 5,
) -> np.ndarray:
    if len(embeddings) < min_frames:
        raise ValueError(f"At least {min_frames} enrollment frames are required")
    normalized = np.stack([normalize_embedding(embedding) for embedding in embeddings])
    return normalize_embedding(np.median(normalized, axis=0))


class MultiFrameVerifier:
    def __init__(self, template: np.ndarray, config: VerificationConfig) -> None:
        self.template = normalize_embedding(template)
        self.config = config

    def evaluate(self, probe_embeddings: list[np.ndarray]) -> GateResult:
        if len(probe_embeddings) < self.config.min_probe_frames:
            return GateResult(
                gate="identity",
                status=GateStatus.FAIL,
                reason_codes=(reason_codes.INSUFFICIENT_VALID_FRAMES,),
                model_version=self.config.model_version,
                threshold_version=self.config.threshold_version,
            )
        similarities = np.array(
            [
                float(np.dot(normalize_embedding(embedding), self.template))
                for embedding in probe_embeddings
            ],
            dtype=np.float32,
        )
        score = float(np.median(similarities))
        passed = score >= self.config.threshold
        return GateResult(
            gate="identity",
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            score=score,
            threshold=self.config.threshold,
            reason_codes=() if passed else (reason_codes.LOW_IDENTITY_SIMILARITY,),
            model_version=self.config.model_version,
            threshold_version=self.config.threshold_version,
        )


class FaceNetEmbedder:
    model_version = "facenet-vggface2-2.6.0"

    def __init__(self, device=None) -> None:
        self.device = device

    def embed(self, images: list[Image.Image]) -> list[np.ndarray]:
        from src.verification.defenses.facenet_embed import get_embedding

        return [get_embedding(image, self.device).numpy() for image in images]
