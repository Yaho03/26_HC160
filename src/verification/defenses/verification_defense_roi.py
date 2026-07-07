"""
ROI-first Defense (Otsu 이진화 기반 얼굴 영역 마스킹)

배경에 분산된 adversarial perturbation을 차단하기 위해 Otsu 이진화로
얼굴 영역만 남기고 배경을 얼굴 영역 평균 픽셀값으로 대체한다.

한계: 얼굴 영역 내 집중 공격(PGD 등)에는 효과 제한적.
배경 / 헤어 / 의류 등 비얼굴 영역에 분산된 perturbation 차단에 효과적.

실행:
    python -m src.verification.defenses.verification_defense_roi \\
        --index  outputs/verification_defense/attack_handoff_jpeg_index.csv \\
        --pkg-root /path/to/pkg \\
        --out-dir outputs/verification_defense
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
from PIL import Image

from src.verification.defenses.verification_defense_base import run_verification_defense


# ── Otsu ROI 마스킹 ───────────────────────────────────────────────────────────

def roi_first_defense(img: Image.Image, morph_kernel: int = 5) -> Image.Image:
    """
    Otsu 이진화로 얼굴 영역 마스크를 생성하고
    배경 픽셀을 얼굴 영역 평균 RGB로 대체한다.

    Args:
        img:           원본 PIL 이미지 (RGB)
        morph_kernel:  마스크 정제용 morphological closing 커널 크기

    Returns:
        배경이 평균값으로 대체된 PIL 이미지
    """
    arr = np.array(img.convert("RGB"), dtype=np.uint8)      # (H, W, 3)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)            # (H, W)

    # Otsu 이진화 — 얼굴(밝은 영역)=255, 배경(어두운 영역)=0
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing으로 마스크 노이즈 제거 (작은 구멍 채우기)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    face_mask = mask > 0  # (H, W) bool

    # 얼굴 영역 평균 RGB 계산
    face_pixels = arr[face_mask]  # (N, 3)
    if len(face_pixels) == 0:
        # 마스크가 비어있으면 원본 반환
        return img
    mean_rgb = face_pixels.mean(axis=0).astype(np.uint8)   # (3,)

    # 배경 대체
    result = arr.copy()
    result[~face_mask] = mean_rgb

    return Image.fromarray(result)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ROI-first Defense (Otsu 이진화)")
    parser.add_argument("--index",       required=True, help="attack_handoff_jpeg_index.csv 경로")
    parser.add_argument("--pkg-root",    required=True, help="패키지 루트 (samples/ 상위)")
    parser.add_argument("--out-dir",     default="outputs/verification_defense", help="결과 저장 디렉터리")
    parser.add_argument("--morph-kernel", type=int, default=5, help="Morphological closing 커널 크기 (기본 5)")
    args = parser.parse_args()

    kernel = args.morph_kernel
    defense_fn = lambda img: roi_first_defense(img, morph_kernel=kernel)

    run_verification_defense(
        defense_name="roi_first",
        defense_fn=defense_fn,
        defense_params={"morph_kernel": kernel},
        index_csv=args.index,
        pkg_root=args.pkg_root,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
