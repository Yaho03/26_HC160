"""
Verification 기반 Temporal Consistency 방어 (시뮬레이션 모드)

단일 이미지에 미세한 augmentation을 N회 적용해 연속 프레임을 시뮬레이션한다.
프레임 간 FaceNet embedding 변화량을 분석하여 비정상 입력을 탐지한다.

탐지 기준:
  - embedding_std < static_threshold  → 정지 이미지 (프린트/화면 재생 의심)
  - mean cosine similarity < threshold → 공격 무력화 (augmentation으로 perturbation 분산)

실행:
    python -m src.verification.defenses.verification_defense_temporal \\
        --index  outputs/verification_defense/attack_handoff_jpeg_index.csv \\
        --pkg-root /path/to/pkg \\
        --n-frames 10 \\
        --out-dir outputs/verification_defense
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import List

import numpy as np
import torch
from PIL import Image, ImageEnhance

from src.verification.defenses.facenet_embed import get_embedding, cosine_similarity


# ── 기본 파라미터 ─────────────────────────────────────────────────────────────

DEFAULT_N_FRAMES      = 10      # 시뮬레이션 프레임 수
DEFAULT_ROT_RANGE     = 5.0     # 회전 범위 ±도
DEFAULT_BRIGHT_RANGE  = 0.15    # 밝기 변화 범위 ±비율
DEFAULT_CONTRAST_RANGE= 0.10    # 대비 변화 범위 ±비율
DEFAULT_STATIC_THRESH = 0.015   # embedding std 임계값 (이하면 정지 이미지)
DEFAULT_THRESHOLD     = 0.47966246581077576


# ── Augmentation ─────────────────────────────────────────────────────────────

def augment_frame(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """단일 이미지에 미세한 랜덤 augmentation을 적용해 한 프레임을 생성한다."""
    # 회전
    angle = rng.uniform(-DEFAULT_ROT_RANGE, DEFAULT_ROT_RANGE)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))

    # 밝기
    factor = 1.0 + rng.uniform(-DEFAULT_BRIGHT_RANGE, DEFAULT_BRIGHT_RANGE)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # 대비
    factor = 1.0 + rng.uniform(-DEFAULT_CONTRAST_RANGE, DEFAULT_CONTRAST_RANGE)
    img = ImageEnhance.Contrast(img).enhance(factor)

    return img


def generate_frames(img: Image.Image, n: int, seed: int = 42) -> List[Image.Image]:
    """원본 이미지로부터 N개의 시뮬레이션 프레임을 생성한다."""
    rng = np.random.default_rng(seed)
    frames = [img]  # 첫 프레임은 원본
    for _ in range(n - 1):
        frames.append(augment_frame(img.copy(), rng))
    return frames


# ── 탐지 핵심 로직 ────────────────────────────────────────────────────────────

def temporal_consistency_check(
    img: Image.Image,
    enroll_img: Image.Image,
    threshold: float = DEFAULT_THRESHOLD,
    n_frames: int = DEFAULT_N_FRAMES,
    static_thresh: float = DEFAULT_STATIC_THRESH,
    device=None,
) -> dict:
    """
    단일 이미지의 temporal consistency를 검사한다.

    Returns:
        {
          "accepted":          bool   최종 인증 통과 여부
          "reject_reason":     str    거부 사유 (없으면 None)
          "mean_similarity":   float  N프레임 평균 similarity
          "embedding_std":     float  프레임 간 embedding 표준편차 평균
          "is_static":         bool   정지 이미지 탐지 여부
          "frame_similarities": list  프레임별 similarity
        }
    """
    frames = generate_frames(img, n_frames)
    enroll_emb = get_embedding(enroll_img, device)

    embeddings = []
    sims = []

    for frame in frames:
        emb = get_embedding(frame, device)
        sim = cosine_similarity(emb, enroll_emb)
        embeddings.append(emb.numpy())
        sims.append(sim)

    emb_array = np.stack(embeddings)  # (N, 512)

    # 프레임 간 embedding 표준편차 (각 차원의 std 평균)
    embedding_std = float(np.mean(np.std(emb_array, axis=0)))

    # 평균 similarity
    mean_sim = float(np.mean(sims))

    # 정지 이미지 탐지
    is_static = embedding_std < static_thresh

    # 최종 판정
    if is_static:
        accepted = False
        reject_reason = "static_image"
    elif mean_sim < threshold:
        accepted = False
        reject_reason = "low_mean_similarity"
    else:
        accepted = True
        reject_reason = None

    return {
        "accepted":           accepted,
        "reject_reason":      reject_reason,
        "mean_similarity":    round(mean_sim, 8),
        "embedding_std":      round(embedding_std, 8),
        "is_static":          is_static,
        "frame_similarities": [round(s, 8) for s in sims],
    }


# ── 평가 루프 ─────────────────────────────────────────────────────────────────

def run_temporal_defense(
    index_csv: str,
    pkg_root: str,
    out_dir: str,
    n_frames: int = DEFAULT_N_FRAMES,
    static_thresh: float = DEFAULT_STATIC_THRESH,
    threshold: float = DEFAULT_THRESHOLD,
    device=None,
) -> str:
    from tqdm import tqdm
    import json

    rows = list(csv.DictReader(open(index_csv)))
    out_rows = []

    for row in tqdm(rows, desc="Temporal Consistency 방어", unit="샘플"):
        adv_path   = os.path.join(pkg_root, row["adv_file"])
        enroll_path = os.path.join(pkg_root, row["target_enroll_file"])
        t0 = time.perf_counter()

        try:
            adv_img    = Image.open(adv_path).convert("RGB")
            enroll_img = Image.open(enroll_path).convert("RGB")
            result = temporal_consistency_check(
                adv_img, enroll_img,
                threshold=threshold,
                n_frames=n_frames,
                static_thresh=static_thresh,
                device=device,
            )
        except Exception as e:
            print(f"[오류] {row['sample_id']}: {e}")
            result = {
                "accepted": None, "reject_reason": "ERROR",
                "mean_similarity": None, "embedding_std": None,
                "is_static": None, "frame_similarities": [],
            }

        elapsed = time.perf_counter() - t0

        accepted_after_attack  = str(row.get("accepted_after_attack", "False")).lower() == "true"
        accepted_after_defense = result["accepted"]
        defense_success = (
            accepted_after_attack and
            accepted_after_defense is not None and
            not accepted_after_defense
        )

        out_rows.append({
            "sample_id":               row["sample_id"],
            "defense":                 "temporal_consistency",
            "defense_params":          json.dumps({"n_frames": n_frames, "static_thresh": static_thresh}),
            "threshold":               threshold,
            "similarity_after_attack": row.get("similarity_after_attack", ""),
            "mean_similarity":         result["mean_similarity"],
            "embedding_std":           result["embedding_std"],
            "is_static":               result["is_static"],
            "reject_reason":           result["reject_reason"],
            "accepted_after_attack":   accepted_after_attack,
            "accepted_after_defense":  accepted_after_defense,
            "attack_success_after_defense": accepted_after_defense,
            "defense_success":         defense_success,
            "defense_time_sec":        round(elapsed, 6),
        })

    # 결과 저장
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "temporal", "verification_defense_temporal.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fieldnames = list(out_rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    # 집계 출력
    total   = len(out_rows)
    n_atk   = sum(1 for r in out_rows if str(r["accepted_after_attack"]) == "True")
    n_def   = sum(1 for r in out_rows if str(r["defense_success"]) == "True")
    n_still = sum(1 for r in out_rows if str(r["attack_success_after_defense"]) == "True")
    n_static = sum(1 for r in out_rows if str(r["is_static"]) == "True")

    print(f"\n완료 — 전체 {total}개")
    print(f"  공격 성공 (방어 전):   {n_atk}개 ({n_atk/total*100:.1f}%)")
    print(f"  정지 이미지 탐지:      {n_static}개 ({n_static/total*100:.1f}%)")
    print(f"  방어 성공:             {n_def}개 ({n_def/total*100:.1f}%)")
    print(f"  방어 후 공격 성공:     {n_still}개 ({n_still/total*100:.1f}%)")
    print(f"  저장: {out_path}")

    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal Consistency 방어 (시뮬레이션)")
    parser.add_argument("--index",         required=True, help="attack_handoff_jpeg_index.csv 경로")
    parser.add_argument("--pkg-root",      required=True, help="패키지 루트 (samples/ 상위)")
    parser.add_argument("--n-frames",      type=int,   default=DEFAULT_N_FRAMES,   help=f"시뮬레이션 프레임 수 (기본 {DEFAULT_N_FRAMES})")
    parser.add_argument("--static-thresh", type=float, default=DEFAULT_STATIC_THRESH, help=f"정지 이미지 탐지 임계값 (기본 {DEFAULT_STATIC_THRESH})")
    parser.add_argument("--out-dir",       default="outputs/verification_defense",  help="결과 저장 디렉터리")
    args = parser.parse_args()

    run_temporal_defense(
        index_csv=args.index,
        pkg_root=args.pkg_root,
        out_dir=args.out_dir,
        n_frames=args.n_frames,
        static_thresh=args.static_thresh,
    )
