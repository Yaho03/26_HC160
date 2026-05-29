"""
Verification 기반 Gaussian Smoothing 방어

adv 이미지에 Gaussian Blur 를 적용한 뒤 FaceNet cosine similarity 로 평가한다.

실행:
    python -m src.defenses.verification_defense_smoothing \\
        --index <attack_handoff_index.csv> \\
        --pkg-root <패키지 루트> \\
        [--radius 3] \\
        [--out-dir outputs/verification_defense]
"""

from __future__ import annotations

import argparse

from PIL import Image, ImageFilter

from src.defenses.verification_defense_base import run_verification_defense

# ── 기본 파라미터 ─────────────────────────────────────────────────────────────
DEFAULT_RADIUS = 3.0


# ── 변환 함수 ─────────────────────────────────────────────────────────────────
def defend(img: Image.Image, radius: float = DEFAULT_RADIUS) -> Image.Image:
    """Gaussian Blur 를 적용해 반환한다."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verification Gaussian Smoothing 방어")
    parser.add_argument("--index",    required=True, help="attack_handoff_index.csv 경로")
    parser.add_argument("--pkg-root", required=True, help="패키지 루트 (samples/ 상위)")
    parser.add_argument("--radius",   type=float, default=DEFAULT_RADIUS, help="Gaussian blur 반경 (기본 3.0)")
    parser.add_argument("--out-dir",  default="outputs/verification_defense", help="결과 저장 디렉터리")
    args = parser.parse_args()

    params = {"radius": args.radius}

    run_verification_defense(
        defense_name="smoothing",
        defense_fn=lambda img: defend(img, radius=args.radius),
        defense_params=params,
        index_csv=args.index,
        pkg_root=args.pkg_root,
        out_dir=args.out_dir,
        img_ext="png",
    )
