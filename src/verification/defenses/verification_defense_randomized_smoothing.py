"""
Randomized Smoothing Defense

입력 이미지에 가우시안 노이즈를 N회 반복 주입하고
N개 FaceNet embedding의 평균 벡터로 cosine similarity를 계산한다.

적대적 perturbation은 특정 방향으로 최적화된 구조를 가지므로
무작위 노이즈를 반복 주입하면 perturbation 효과가 분산되어 평균적으로 감소한다.

Certified defense 근거: Cohen et al. (2019) "Certified Adversarial Robustness via Randomized Smoothing"
  - sigma=0.25 수준에서 L2 반경 약 0.25 이내 공격에 대한 certifiable guarantee

실행:
    python -m src.verification.defenses.verification_defense_randomized_smoothing \\
        --index  outputs/verification_defense/attack_handoff_jpeg_index.csv \\
        --pkg-root /path/to/pkg \\
        --n-samples 50 \\
        --sigma 0.05 \\
        --out-dir outputs/verification_defense
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.verification.defenses.facenet_embed import get_embedding, cosine_similarity
from src.verification.defenses.verification_defense_base import THRESHOLD


# ── 기본 파라미터 ─────────────────────────────────────────────────────────────

DEFAULT_N_SAMPLES = 50       # 노이즈 주입 반복 횟수
DEFAULT_SIGMA     = 0.05     # 가우시안 노이즈 표준편차 (픽셀 0~255 스케일 기준 약 12.75)


# ── 핵심 로직 ─────────────────────────────────────────────────────────────────

def add_gaussian_noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    """
    이미지에 가우시안 노이즈를 추가한다.

    Args:
        img:   원본 PIL 이미지 (RGB)
        sigma: 노이즈 표준편차 (0~1 스케일, 즉 sigma=0.05 → 픽셀 ±12.75)
        rng:   numpy random generator (재현성 보장)

    Returns:
        노이즈가 추가된 PIL 이미지 (픽셀값 0~255 클리핑)
    """
    arr = np.array(img, dtype=np.float32) / 255.0   # [0, 1] 스케일
    noise = rng.normal(0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def randomized_smoothing_check(
    adv_img: Image.Image,
    enroll_img: Image.Image,
    threshold: float = THRESHOLD,
    n_samples: int = DEFAULT_N_SAMPLES,
    sigma: float = DEFAULT_SIGMA,
    device=None,
) -> dict:
    """
    N개 노이즈 샘플의 평균 embedding으로 cosine similarity를 계산한다.

    Returns:
        {
          "accepted":        bool   최종 인증 통과 여부
          "mean_similarity": float  평균 similarity
          "std_similarity":  float  similarity 표준편차 (노이즈 분산도)
          "n_accepted":      int    개별 샘플 중 accept된 수
        }
    """
    rng = np.random.default_rng(42)
    enroll_emb = get_embedding(enroll_img, device)  # (512,)

    embeddings: List[torch.Tensor] = []
    sims: List[float] = []

    for _ in range(n_samples):
        noisy_img = add_gaussian_noise(adv_img, sigma, rng)
        emb = get_embedding(noisy_img, device)  # (512,)
        embeddings.append(emb)
        sims.append(cosine_similarity(emb, enroll_emb))

    # 평균 embedding (L2 재정규화)
    mean_emb = torch.stack(embeddings).mean(dim=0)
    mean_emb = F.normalize(mean_emb.unsqueeze(0), p=2, dim=1).squeeze(0)
    mean_sim_from_avg_emb = cosine_similarity(mean_emb, enroll_emb)

    mean_sim = float(np.mean(sims))
    std_sim  = float(np.std(sims))
    n_accepted = sum(1 for s in sims if s >= threshold)

    return {
        "accepted":               mean_sim_from_avg_emb >= threshold,
        "mean_similarity":        round(mean_sim_from_avg_emb, 8),
        "raw_mean_similarity":    round(mean_sim, 8),
        "std_similarity":         round(std_sim, 8),
        "n_accepted":             n_accepted,
        "accept_rate":            round(n_accepted / n_samples, 4),
    }


# ── 평가 루프 ─────────────────────────────────────────────────────────────────

def run_randomized_smoothing_defense(
    index_csv: str,
    pkg_root: str,
    out_dir: str,
    n_samples: int = DEFAULT_N_SAMPLES,
    sigma: float = DEFAULT_SIGMA,
    threshold: float = THRESHOLD,
    device=None,
) -> str:
    from tqdm import tqdm

    rows = list(csv.DictReader(open(index_csv)))
    out_rows = []

    for row in tqdm(rows, desc="Randomized Smoothing 방어", unit="샘플"):
        adv_path    = os.path.join(pkg_root, row["adv_file"])
        enroll_path = os.path.join(pkg_root, row["target_enroll_file"])
        t0 = time.perf_counter()

        try:
            adv_img    = Image.open(adv_path).convert("RGB")
            enroll_img = Image.open(enroll_path).convert("RGB")
            result = randomized_smoothing_check(
                adv_img, enroll_img,
                threshold=threshold,
                n_samples=n_samples,
                sigma=sigma,
                device=device,
            )
        except Exception as e:
            print(f"[오류] {row['sample_id']}: {e}")
            result = {
                "accepted": None, "mean_similarity": None,
                "raw_mean_similarity": None, "std_similarity": None,
                "n_accepted": None, "accept_rate": None,
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
            "sample_id":                   row["sample_id"],
            "defense":                     "randomized_smoothing",
            "defense_params":              json.dumps({"n_samples": n_samples, "sigma": sigma}),
            "threshold":                   threshold,
            "similarity_after_attack":     row.get("similarity_after_attack", ""),
            "mean_similarity":             result["mean_similarity"],
            "raw_mean_similarity":         result["raw_mean_similarity"],
            "std_similarity":              result["std_similarity"],
            "n_accepted":                  result["n_accepted"],
            "accept_rate":                 result["accept_rate"],
            "accepted_after_attack":       accepted_after_attack,
            "accepted_after_defense":      accepted_after_defense,
            "attack_success_after_defense": accepted_after_defense,
            "defense_success":             defense_success,
            "defense_time_sec":            round(elapsed, 6),
        })

    # 저장
    import pathlib
    out_path = pathlib.Path(out_dir) / "randomized_smoothing" / "verification_defense_randomized_smoothing.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(out_rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    # 집계
    total   = len(out_rows)
    n_atk   = sum(1 for r in out_rows if str(r["accepted_after_attack"]) == "True")
    n_def   = sum(1 for r in out_rows if str(r["defense_success"]) == "True")
    n_still = sum(1 for r in out_rows if str(r["attack_success_after_defense"]) == "True")

    print(f"\n완료 — 전체 {total}개")
    print(f"  공격 성공 (방어 전):   {n_atk}개 ({n_atk/total*100:.1f}%)")
    print(f"  방어 성공:             {n_def}개 ({n_def/total*100:.1f}%)")
    print(f"  방어 후 공격 성공:     {n_still}개 ({n_still/total*100:.1f}%)")
    print(f"  저장: {out_path}")

    return str(out_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomized Smoothing Defense")
    parser.add_argument("--index",     required=True, help="attack_handoff_jpeg_index.csv 경로")
    parser.add_argument("--pkg-root",  required=True, help="패키지 루트 (samples/ 상위)")
    parser.add_argument("--n-samples", type=int,   default=DEFAULT_N_SAMPLES, help=f"노이즈 주입 반복 횟수 (기본 {DEFAULT_N_SAMPLES})")
    parser.add_argument("--sigma",     type=float, default=DEFAULT_SIGMA,     help=f"가우시안 노이즈 표준편차 (기본 {DEFAULT_SIGMA})")
    parser.add_argument("--out-dir",   default="outputs/verification_defense", help="결과 저장 디렉터리")
    args = parser.parse_args()

    run_randomized_smoothing_defense(
        index_csv=args.index,
        pkg_root=args.pkg_root,
        out_dir=args.out_dir,
        n_samples=args.n_samples,
        sigma=args.sigma,
    )
