"""Apply ROI-first masking defense to adversarial face images.

This baseline consumes `outputs/attacks/attack_index.csv`, detects the face
region in `adv_file` (Haar cascade, falling back to a centered ellipse when no
face is found), suppresses everything outside that region (blackout or
attenuate), re-runs the trained face classifier, and writes a result CSV that
can be joined back to attack metadata by `sample_id`.

The idea is to shrink the attack surface: perturbations injected into
background pixels outside the face ROI never reach the classifier.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.defenses._pipeline import run_defense_pipeline


def _ascii_safe_cascade_path(cascade_filename: str) -> str:
    """Work around an OpenCV/Windows bug where CascadeClassifier silently
    fails to open files under a path containing non-ASCII characters (e.g. a
    Windows username with Korean characters). Caches an ASCII-only copy under
    a fixed system directory and reuses it across runs."""
    src = Path(cv2.data.haarcascades) / cascade_filename
    if str(src).isascii() or sys.platform != "win32":
        return str(src)

    cache_dir = Path(tempfile.gettempdir()).anchor + "ProgramData/hanium_aml_cache"
    cache_path = Path(cache_dir) / cascade_filename
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, cache_path)
    return str(cache_path)


_FACE_CASCADE = cv2.CascadeClassifier(_ascii_safe_cascade_path("haarcascade_frontalface_default.xml"))


def detect_face_bbox(image_rgb: np.ndarray) -> tuple[int, int, int, int]:
    """Return (x, y, w, h) of the largest detected face, or a centered fallback box."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(int(gray.shape[1] * 0.2),) * 2
    )
    if len(faces) > 0:
        return max(faces, key=lambda box: box[2] * box[3])

    h, w = gray.shape
    box_w, box_h = int(w * 0.7), int(h * 0.7)
    return ((w - box_w) // 2, (h - box_h) // 2, box_w, box_h)


def build_roi_mask(image_rgb: np.ndarray, margin: float) -> np.ndarray:
    """Elliptical soft mask (1.0 inside ROI, 0.0 outside) sized to the face bbox."""
    h, w = image_rgb.shape[:2]
    x, y, bw, bh = detect_face_bbox(image_rgb)
    cx, cy = x + bw / 2, y + bh / 2
    rx, ry = (bw / 2) * (1 + margin), (bh / 2) * (1 + margin)

    yy, xx = np.mgrid[0:h, 0:w]
    dist = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2
    return (dist <= 1.0).astype(np.float32)


def apply_roi_mask(image: Image.Image, mode: str, margin: float, attenuate_factor: float) -> Image.Image:
    array = np.array(image.convert("RGB"))
    mask = build_roi_mask(array, margin)[..., None]

    if mode == "blackout":
        background = np.zeros_like(array, dtype=np.float32)
    elif mode == "attenuate":
        background = array.astype(np.float32) * attenuate_factor
    else:
        raise ValueError(f"Unknown ROI mode: {mode}")

    blended = array.astype(np.float32) * mask + background * (1 - mask)
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI-first masking defense baseline.")
    parser.add_argument("--attack-index", type=Path, default=Path("outputs/attacks/attack_index.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/face_resnet50_lfw10/best.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/defenses/roi"))
    parser.add_argument("--mode", type=str, default="attenuate", choices=["blackout", "attenuate"])
    parser.add_argument("--attenuate-factor", type=float, default=0.3, help="Used when --mode attenuate.")
    parser.add_argument("--margin", type=float, default=0.15, help="ROI ellipse expansion ratio around detected face bbox.")
    parser.add_argument("--limit", type=int, default=0, help="0 means use all filtered rows.")
    parser.add_argument("--only-success-on-clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attack-family", type=str, default="", help="Optional filter: fgsm, pgd, square, jsma, zoo.")
    args = parser.parse_args()

    if args.mode == "attenuate" and not 0 <= args.attenuate_factor <= 1:
        raise ValueError("--attenuate-factor must be between 0 and 1")

    run_defense_pipeline(
        attack_index_path=args.attack_index,
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        defense_name="roi_first",
        defense_params={"mode": args.mode, "margin": args.margin, "attenuate_factor": args.attenuate_factor},
        transform=lambda image: apply_roi_mask(image, args.mode, args.margin, args.attenuate_factor),
        image_out_dir=args.out_dir / args.mode / "images",
        image_filename=lambda sample_id: f"{sample_id}_roi_{args.mode}.png",
        result_path=args.out_dir / f"roi_results_{args.mode}.csv",
        only_success_on_clean=args.only_success_on_clean,
        attack_family=args.attack_family,
        limit=args.limit,
        progress_desc=f"roi {args.mode}",
    )


if __name__ == "__main__":
    main()
