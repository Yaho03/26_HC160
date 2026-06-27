"""Apply bit-depth reduction defense to adversarial face images.

This baseline consumes `outputs/attacks/attack_index.csv`, quantizes the color
levels of `adv_file`, re-runs the trained face classifier, and writes a result
CSV that can be joined back to attack metadata by `sample_id`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.defenses._pipeline import run_defense_pipeline


def reduce_bit_depth(image: Image.Image, bits: int) -> Image.Image:
    tensor = transforms.ToTensor()(image.convert("RGB"))
    levels = 2 ** bits
    quantized = torch.round(tensor * (levels - 1)) / (levels - 1)
    return transforms.ToPILImage()(quantized.clamp(0, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Bit-depth reduction defense baseline.")
    parser.add_argument("--attack-index", type=Path, default=Path("outputs/attacks/attack_index.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/face_resnet50_lfw10/best.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/defenses/bitdepth"))
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="0 means use all filtered rows.")
    parser.add_argument("--only-success-on-clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attack-family", type=str, default="", help="Optional filter: fgsm, pgd, square, jsma, zoo.")
    args = parser.parse_args()

    if not 1 <= args.bits <= 8:
        raise ValueError("--bits must be between 1 and 8")

    run_defense_pipeline(
        attack_index_path=args.attack_index,
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        defense_name="bit_depth",
        defense_params={"bits": args.bits},
        transform=lambda image: reduce_bit_depth(image, args.bits),
        image_out_dir=args.out_dir / f"{args.bits}bit" / "images",
        image_filename=lambda sample_id: f"{sample_id}_bitdepth_{args.bits}bit.png",
        result_path=args.out_dir / f"bitdepth_results_{args.bits}bit.csv",
        only_success_on_clean=args.only_success_on_clean,
        attack_family=args.attack_family,
        limit=args.limit,
        progress_desc=f"bitdepth {args.bits}bit",
    )


if __name__ == "__main__":
    main()
