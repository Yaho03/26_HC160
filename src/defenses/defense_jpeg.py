"""Apply JPEG recompression defense to adversarial face images.

This baseline consumes `outputs/attacks/attack_index.csv`, applies JPEG
recompression to `adv_file`, re-runs the trained face classifier, and writes a
result CSV that can be joined back to attack metadata by `sample_id`.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

from PIL import Image

from src.defenses._pipeline import run_defense_pipeline


def jpeg_recompress(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=False)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="JPEG recompression defense baseline.")
    parser.add_argument("--attack-index", type=Path, default=Path("outputs/attacks/attack_index.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/face_resnet50_lfw10/best.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/defenses/jpeg"))
    parser.add_argument("--quality", type=int, default=75)
    parser.add_argument("--limit", type=int, default=0, help="0 means use all filtered rows.")
    parser.add_argument("--only-success-on-clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attack-family", type=str, default="", help="Optional filter: fgsm, pgd, square, jsma.")
    args = parser.parse_args()

    if not 1 <= args.quality <= 100:
        raise ValueError("--quality must be between 1 and 100")

    run_defense_pipeline(
        attack_index_path=args.attack_index,
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        defense_name="jpeg",
        defense_params={"quality": args.quality},
        transform=lambda image: jpeg_recompress(image, args.quality),
        image_out_dir=args.out_dir / f"q{args.quality}" / "images",
        image_filename=lambda sample_id: f"{sample_id}_jpeg_q{args.quality}.jpg",
        result_path=args.out_dir / f"jpeg_results_q{args.quality}.csv",
        only_success_on_clean=args.only_success_on_clean,
        attack_family=args.attack_family,
        limit=args.limit,
        progress_desc=f"jpeg q={args.quality}",
        save_format="JPEG",
        save_kwargs={"quality": 95},
    )


if __name__ == "__main__":
    main()
