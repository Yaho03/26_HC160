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
        """
        이미지 목록을 한 배치로 forward한다.

        이 크기에서 FaceNet forward는 연산이 아니라 호출당 고정 비용에 묶여 있다.
        한 장씩 호출하면 그 비용을 이미지 수만큼 낸다. PERF-001 실측에서 mps 기준
        9장 개별 호출 602.85 ms, 같은 9장 배치 1회 77.86 ms였다.
        docs/experiments/PERF-001-detector-latency.md 5.2절 참조.

        전처리 계약은 그대로다. preprocess를 공유하므로 160x160 bilinear와
        (x-127.5)/128.0 정규화가 같다. 반환 dtype도 float32를 유지한다.
        get_embedding이 torch 텐서를 float32로 돌려주던 것과 맞춘다.
        """
        import torch
        import torch.nn.functional as F

        from src.verification.defenses.facenet_embed import get_model, preprocess

        if not images:
            return []

        model, device = get_model(self.device)
        batch = torch.cat([preprocess(image) for image in images]).to(device)
        with torch.no_grad():
            embeddings = model(batch)
        embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
        return [vector for vector in embeddings]
