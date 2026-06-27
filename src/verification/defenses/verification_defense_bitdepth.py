"""
Verification 기반 Bit-depth 축소 방어

adv 이미지의 색상 비트 수를 줄여 perturbation 을 제거한 뒤
FaceNet cosine similarity 로 평가한다.

실행:
    python -m src.verification.defenses.verification_defense_bitdepth \\
        --index <attack_handoff_index.csv> \\
        --pkg-root <패키지 루트> \\
        [--bits 4] \\
        [--out-dir outputs/verification_defense]
"""

from __future__ import annotations

import argparse

import numpy as np
from PIL import Image

from src.verification.defenses.verification_defense_base import run_verification_defense

# ── 기본 파라미터 ─────────────────────────────────────────────────────────────
DEFAULT_BITS = 4


# ── 변환 함수 ─────────────────────────────────────────────────────────────────
def defend(img: Image.Image, bits: int = DEFAULT_BITS) -> Image.Image:
    """비트 깊이를 축소해 반환한다. (8-bit → bits-bit)"""
    arr   = np.array(img, dtype=np.uint8)
    shift = 8 - bits
    arr   = (arr >> shift) << shift
    return Image.fromarray(arr)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verification Bit-depth 축소 방어")
    parser.add_argument("--index",    required=True, help="attack_handoff_index.csv 경로")
    parser.add_argument("--pkg-root", required=True, help="패키지 루트 (samples/ 상위)")
    parser.add_argument("--bits",     type=int, default=DEFAULT_BITS, help="축소할 비트 수 (기본 4)")
    parser.add_argument("--out-dir",  default="outputs/verification_defense", help="결과 저장 디렉터리")
    args = parser.parse_args()

    params = {"bits": args.bits}

    run_verification_defense(
        defense_name="bitdepth",
        defense_fn=lambda img: defend(img, bits=args.bits),
        defense_params=params,
        index_csv=args.index,
        pkg_root=args.pkg_root,
        out_dir=args.out_dir,
        img_ext="png",
    )
