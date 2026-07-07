"""
Feature Squeezing Defense (4단계 - 사후 포렌식 탐지)

원본 이미지와 squeezed 이미지의 FaceNet similarity 차이로 공격을 탐지한다.
적대적 perturbation은 특정 픽셀 구조에 최적화되어 있어 squeezing 후 similarity가
크게 변하는 반면, 정상 이미지는 squeezing 전후 similarity 변화가 작다.

탐지 목적 (방어 차단이 아닌 risk_score 산출 → 대시보드 전송).

Squeezer 3종:
  - low_resolution: 32×32 축소 후 160×160 복원
  - color_depth:    4비트 양자화 (256색 → 16색)
  - median_filter:  3×3 미디언 필터

실행:
    python -m src.verification.defenses.verification_defense_feature_squeezing \\
        --index  outputs/verification_defense/attack_handoff_jpeg_index.csv \\
        --pkg-root /path/to/pkg \\
        --out-dir outputs/verification_defense
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.verification.defenses.facenet_embed import get_embedding, cosine_similarity
from src.verification.defenses.verification_defense_base import THRESHOLD


# ── Squeezer 함수 ─────────────────────────────────────────────────────────────

def squeeze_low_resolution(img: Image.Image, low_res: int = 32) -> Image.Image:
    """32×32로 축소 후 원래 크기로 복원 — 고주파 perturbation 제거."""
    orig_size = img.size
    small = img.resize((low_res, low_res), Image.BILINEAR)
    return small.resize(orig_size, Image.BILINEAR)


def squeeze_color_depth(img: Image.Image, bits: int = 4) -> Image.Image:
    """비트 심도 축소 — 미세한 픽셀 변화 소거."""
    arr = np.array(img, dtype=np.float32)
    levels = 2 ** bits
    arr = np.floor(arr / 256.0 * levels) / levels * 256.0
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def squeeze_median_filter(img: Image.Image, kernel: int = 3) -> Image.Image:
    """미디언 필터 — 국소 perturbation 평활화."""
    arr = np.array(img, dtype=np.uint8)
    filtered = cv2.medianBlur(arr, kernel)
    return Image.fromarray(filtered)


SQUEEZERS = {
    "low_resolution": squeeze_low_resolution,
    "color_depth":    squeeze_color_depth,
    "median_filter":  squeeze_median_filter,
}

DEFAULT_DETECTION_THRESHOLD = 0.05  # sim_diff 이 이상이면 공격 탐지


# ── 단일 샘플 분석 ────────────────────────────────────────────────────────────

def feature_squeezing_check(
    adv_img: Image.Image,
    enroll_img: Image.Image,
    detection_threshold: float = DEFAULT_DETECTION_THRESHOLD,
    device=None,
) -> dict:
    """
    3종 squeezer별 similarity 차이를 계산하고 공격 탐지 여부를 반환한다.

    Returns:
        {
          "is_attack_detected": bool   최소 1개 squeezer에서 탐지 여부
          "max_sim_diff":       float  최대 similarity 차이
          "risk_score_add":     int    탐지 기반 추가 risk score (0~30)
          "squeezers":          dict   squeezer별 상세 결과
        }
    """
    enroll_emb  = get_embedding(enroll_img, device)
    sim_original = cosine_similarity(get_embedding(adv_img, device), enroll_emb)

    results = {}
    for name, fn in SQUEEZERS.items():
        squeezed     = fn(adv_img.copy())
        sim_squeezed = cosine_similarity(get_embedding(squeezed, device), enroll_emb)
        sim_diff     = abs(sim_original - sim_squeezed)
        detected     = sim_diff >= detection_threshold
        results[name] = {
            "sim_original":     round(sim_original, 8),
            "sim_squeezed":     round(sim_squeezed, 8),
            "sim_diff":         round(sim_diff, 8),
            "is_attack_detected": detected,
            "detection_threshold": detection_threshold,
        }

    n_detected   = sum(1 for r in results.values() if r["is_attack_detected"])
    max_sim_diff = max(r["sim_diff"] for r in results.values())

    # risk_score 추가분: 탐지 수 × 10 (최대 30)
    risk_score_add = min(n_detected * 10, 30)

    return {
        "is_attack_detected": n_detected > 0,
        "n_squeezers_detected": n_detected,
        "max_sim_diff":       round(max_sim_diff, 8),
        "risk_score_add":     risk_score_add,
        "squeezers":          results,
    }


# ── 평가 루프 ─────────────────────────────────────────────────────────────────

def run_feature_squeezing(
    index_csv: str,
    pkg_root: str,
    out_dir: str,
    detection_threshold: float = DEFAULT_DETECTION_THRESHOLD,
    device=None,
) -> str:
    from tqdm import tqdm

    rows = list(csv.DictReader(open(index_csv)))
    out_rows = []

    for row in tqdm(rows, desc="Feature Squeezing 탐지", unit="샘플"):
        adv_path    = os.path.join(pkg_root, row["adv_file"])
        enroll_path = os.path.join(pkg_root, row["target_enroll_file"])
        t0 = time.perf_counter()

        try:
            adv_img    = Image.open(adv_path).convert("RGB")
            enroll_img = Image.open(enroll_path).convert("RGB")
            result = feature_squeezing_check(
                adv_img, enroll_img,
                detection_threshold=detection_threshold,
                device=device,
            )
        except Exception as e:
            print(f"[오류] {row['sample_id']}: {e}")
            result = {
                "is_attack_detected": None, "n_squeezers_detected": None,
                "max_sim_diff": None, "risk_score_add": None,
                "squeezers": {},
            }

        elapsed = time.perf_counter() - t0
        accepted_after_attack = str(row.get("accepted_after_attack", "False")).lower() == "true"

        sq = result["squeezers"]
        out_rows.append({
            "sample_id":                row["sample_id"],
            "defense":                  "feature_squeezing",
            "detection_threshold":      detection_threshold,
            "accepted_after_attack":    accepted_after_attack,
            "is_attack_detected":       result["is_attack_detected"],
            "n_squeezers_detected":     result["n_squeezers_detected"],
            "max_sim_diff":             result["max_sim_diff"],
            "risk_score_add":           result["risk_score_add"],
            # low_resolution
            "lr_sim_original":   sq.get("low_resolution", {}).get("sim_original"),
            "lr_sim_squeezed":   sq.get("low_resolution", {}).get("sim_squeezed"),
            "lr_sim_diff":       sq.get("low_resolution", {}).get("sim_diff"),
            "lr_detected":       sq.get("low_resolution", {}).get("is_attack_detected"),
            # color_depth
            "cd_sim_original":   sq.get("color_depth", {}).get("sim_original"),
            "cd_sim_squeezed":   sq.get("color_depth", {}).get("sim_squeezed"),
            "cd_sim_diff":       sq.get("color_depth", {}).get("sim_diff"),
            "cd_detected":       sq.get("color_depth", {}).get("is_attack_detected"),
            # median_filter
            "mf_sim_original":   sq.get("median_filter", {}).get("sim_original"),
            "mf_sim_squeezed":   sq.get("median_filter", {}).get("sim_squeezed"),
            "mf_sim_diff":       sq.get("median_filter", {}).get("sim_diff"),
            "mf_detected":       sq.get("median_filter", {}).get("is_attack_detected"),
            "defense_time_sec":  round(elapsed, 6),
        })

    # 저장
    out_path = Path(out_dir) / "feature_squeezing" / "verification_defense_feature_squeezing.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    # 집계
    total      = len(out_rows)
    n_detected = sum(1 for r in out_rows if r["is_attack_detected"] is True)
    n_atk      = sum(1 for r in out_rows if r["accepted_after_attack"] is True)
    n_detected_in_atk = sum(
        1 for r in out_rows
        if r["accepted_after_attack"] is True and r["is_attack_detected"] is True
    )

    print(f"\n완료 — 전체 {total}개")
    print(f"  공격 성공 샘플:         {n_atk}개")
    print(f"  전체 탐지:              {n_detected}개 ({n_detected/total*100:.1f}%)")
    print(f"  공격 성공 중 탐지:      {n_detected_in_atk}/{n_atk} ({n_detected_in_atk/n_atk*100:.1f}%)" if n_atk else "")

    # squeezer별 탐지율
    for prefix, name in [("lr", "low_resolution"), ("cd", "color_depth"), ("mf", "median_filter")]:
        n = sum(1 for r in out_rows if r.get(f"{prefix}_detected") is True)
        print(f"  {name:<20}: {n}개 ({n/total*100:.1f}%)")

    print(f"  저장: {out_path}")
    return str(out_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Squeezing 탐지 (4단계)")
    parser.add_argument("--index",                required=True)
    parser.add_argument("--pkg-root",             required=True)
    parser.add_argument("--out-dir",              default="outputs/verification_defense")
    parser.add_argument("--detection-threshold",  type=float, default=DEFAULT_DETECTION_THRESHOLD)
    args = parser.parse_args()

    run_feature_squeezing(
        index_csv=args.index,
        pkg_root=args.pkg_root,
        out_dir=args.out_dir,
        detection_threshold=args.detection_threshold,
    )
