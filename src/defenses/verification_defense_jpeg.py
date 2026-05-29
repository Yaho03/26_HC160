"""
Verification 기반 JPEG 압축 방어

adv 이미지에 JPEG 압축을 적용한 뒤 FaceNet cosine similarity 로 평가한다.

실행:
    python -m src.defenses.verification_defense_jpeg \\
        --index <attack_handoff_index.csv> \\
        --pkg-root <패키지 루트> \\
        [--quality 75] \\
        [--out-dir outputs/verification_defense]
"""

from __future__ import annotations

import argparse
from io import BytesIO

from PIL import Image

from src.defenses.verification_defense_base import run_verification_defense

# ── 기본 파라미터 ─────────────────────────────────────────────────────────────
DEFAULT_QUALITY = 75


# ── 변환 함수 ─────────────────────────────────────────────────────────────────
def defend(img: Image.Image, quality: int = DEFAULT_QUALITY) -> Image.Image:
    """JPEG 압축을 적용해 반환한다."""
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verification JPEG 방어")
    parser.add_argument("--index",    required=True, help="attack_handoff_index.csv 경로")
    parser.add_argument("--pkg-root", required=True, help="패키지 루트 (samples/ 상위)")
    parser.add_argument("--quality",  type=int, default=DEFAULT_QUALITY, help="JPEG quality (기본 75)")
    parser.add_argument("--out-dir",  default="outputs/verification_defense", help="결과 저장 디렉터리")
    args = parser.parse_args()

    params = {"quality": args.quality}

    run_verification_defense(
        defense_name="jpeg",
        defense_fn=lambda img: defend(img, quality=args.quality),
        defense_params=params,
        index_csv=args.index,
        pkg_root=args.pkg_root,
        out_dir=args.out_dir,
        img_ext="jpg",
    )
