"""Apply Gaussian smoothing defense to adversarial face images.

This baseline consumes `outputs/attacks/attack_index.csv`, applies Gaussian
blur to `adv_file`, re-runs the trained face classifier, and writes a result
CSV that can be joined back to attack metadata by `sample_id`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageFilter

from src.defenses._pipeline import run_defense_pipeline


def gaussian_smooth(image: Image.Image, radius: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=radius)).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gaussian smoothing defense baseline.")
    parser.add_argument("--attack-index", type=Path, default=Path("outputs/attacks/attack_index.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/face_resnet50_lfw10/best.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/defenses/smoothing"))
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0, help="0 means use all filtered rows.")
    parser.add_argument("--only-success-on-clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attack-family", type=str, default="", help="Optional filter: fgsm, pgd, square, jsma, zoo.")
    args = parser.parse_args()

    if args.radius <= 0:
        raise ValueError("--radius must be positive")

    radius_tag = str(args.radius).replace(".", "p")

    run_defense_pipeline(
        attack_index_path=args.attack_index,
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        defense_name="gaussian_smoothing",
        defense_params={"radius": args.radius},
        transform=lambda image: gaussian_smooth(image, args.radius),
        image_out_dir=args.out_dir / f"r{radius_tag}" / "images",
        image_filename=lambda sample_id: f"{sample_id}_smoothing_r{radius_tag}.png",
        result_path=args.out_dir / f"smoothing_results_r{radius_tag}.csv",
        only_success_on_clean=args.only_success_on_clean,
        attack_family=args.attack_family,
        limit=args.limit,
        progress_desc=f"smoothing r={args.radius}",
    )


if __name__ == "__main__":
    main()
