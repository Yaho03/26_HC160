"""
FaceNet 임베딩 유틸리티

InceptionResnetV1 (pretrained='vggface2') 를 로드하고,
이미지 → 512-d 임베딩 변환 + cosine similarity 계산을 제공한다.

전처리: 공격팀과 동일하게
  - 160×160 resize (BILINEAR)
  - float32 변환 후 (x - 127.5) / 128.0 정규화
  - CHW 순서, 배치 차원 추가
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from typing import Optional, Tuple


# ── 모델 싱글톤 ──────────────────────────────────────────────────────────────

_model: Optional[InceptionResnetV1] = None
_device: Optional[torch.device] = None


def select_device(device=None) -> torch.device:
    """
    사용할 장치를 고른다. Apple Silicon에서 CPU로 떨어지면 평가가 몇 배 느려진다.

    명시한 값이 있으면 그대로 쓴다. 없으면 CUDA, MPS, CPU 순으로 고른다.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model(device=None) -> Tuple[InceptionResnetV1, torch.device]:
    """모델을 최초 1회만 로드하고 이후 재사용한다."""
    global _model, _device

    if _model is None:
        resolved = select_device(device)
        _model = InceptionResnetV1(pretrained="vggface2").eval().to(resolved)
        _device = resolved

    return _model, _device


# ── 전처리 ───────────────────────────────────────────────────────────────────

def preprocess(img: Image.Image) -> torch.Tensor:
    """
    PIL Image → (1, 3, 160, 160) float32 텐서
    정규화: (pixel - 127.5) / 128.0
    """
    img = img.convert("RGB").resize((160, 160), Image.BILINEAR)
    import numpy as np
    arr = np.array(img, dtype=np.float32)          # (160, 160, 3)
    arr = (arr - 127.5) / 128.0                    # 공격팀과 동일한 정규화
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, 160, 160)
    return tensor.unsqueeze(0)                      # (1, 3, 160, 160)


# ── 임베딩 추출 ──────────────────────────────────────────────────────────────

def get_embedding(img: Image.Image, device=None) -> torch.Tensor:
    """
    PIL Image → 정규화된 512-d 임베딩 벡터 (L2 norm = 1)
    반환: (512,) CPU 텐서
    """
    model, dev = get_model(device)
    tensor = preprocess(img).to(dev)
    with torch.no_grad():
        emb = model(tensor)          # (1, 512)
    emb = F.normalize(emb, p=2, dim=1)
    return emb.squeeze(0).cpu()      # (512,)


def embed_batch(images, device=None):
    """
    PIL Image n장 → (n, 512) L2 정규화 임베딩 행렬 (numpy)

    한 번의 forward로 전부 처리한다. 변환본을 하나씩 호출하면 실시간 기록이
    불가능하므로 계측 경로는 이 함수를 쓴다.
    """
    import numpy as np

    model, dev = get_model(device)
    batch = torch.cat([preprocess(image) for image in images]).to(dev)
    with torch.no_grad():
        embeddings = model(batch)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy().astype(np.float64)


class FaceNetBatchEmbedder:
    """squeeze_probe.BatchEmbedder 계약의 FaceNet 구현."""

    def __init__(self, device=None) -> None:
        self.device = device

    def embed_batch(self, images):
        return embed_batch(images, self.device)


def get_embedding_from_path(path: str, device=None) -> torch.Tensor:
    """파일 경로에서 바로 임베딩을 추출한다."""
    return get_embedding(Image.open(path), device)


# ── Cosine Similarity ────────────────────────────────────────────────────────

def cosine_similarity(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
    """
    두 임베딩 벡터의 cosine similarity를 반환한다.
    입력: L2 정규화된 (512,) 텐서 두 개
    """
    return float(F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item())


def similarity_from_paths(path_a: str, path_b: str, device=None) -> float:
    """두 이미지 경로를 받아 cosine similarity를 바로 반환한다."""
    emb_a = get_embedding_from_path(path_a, device)
    emb_b = get_embedding_from_path(path_b, device)
    return cosine_similarity(emb_a, emb_b)


# ── 정합성 검증 ──────────────────────────────────────────────────────────────

def verify_against_index(index_csv: str, pkg_root: str, n: int = 5, device=None) -> None:
    """
    attack_handoff_index.csv 의 similarity_after_attack 값과
    우리가 다시 계산한 값을 n 샘플 비교해 전처리 정합성을 확인한다.

    Args:
        index_csv: attack_handoff_index.csv 경로
        pkg_root:  samples/ 폴더가 있는 패키지 루트 경로
        n:         비교할 샘플 수
    """
    import csv, os

    print(f"{'sample_id':<20} {'expected':>10} {'ours':>10} {'diff':>8}")
    print("-" * 52)

    rows = list(csv.DictReader(open(index_csv)))[:n]
    for row in rows:
        adv_path    = os.path.join(pkg_root, row["adv_file"])
        enroll_path = os.path.join(pkg_root, row["target_enroll_file"])
        expected    = float(row["similarity_after_attack"])
        ours        = similarity_from_paths(adv_path, enroll_path, device)
        diff        = abs(expected - ours)
        ok          = "✅" if diff < 0.01 else "❌"
        print(f"{row['sample_id']:<20} {expected:>10.6f} {ours:>10.6f} {diff:>8.6f} {ok}")


# ── CLI (간단 테스트) ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FaceNet 임베딩 정합성 검증")
    parser.add_argument("--index",    required=True, help="attack_handoff_index.csv 경로")
    parser.add_argument("--pkg-root", required=True, help="패키지 루트 (samples/ 폴더 상위)")
    parser.add_argument("--n",        type=int, default=5, help="검증할 샘플 수")
    args = parser.parse_args()

    verify_against_index(args.index, args.pkg_root, n=args.n)
