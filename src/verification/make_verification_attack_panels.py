"""Create visual panels for FaceNet verification attack handoff samples."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PANEL_COLUMNS = [
    ("source_file", "source"),
    ("target_enroll_file", "target enroll"),
    ("adv_file", "adversarial"),
    ("perturbation_file", "perturbation"),
]


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except ValueError:
        return default


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_image(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: str = "black") -> None:
    try:
        font = ImageFont.truetype("Arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text(xy, text, fill=fill, font=font)


def panel_for_row(row: dict[str, str], package_dir: Path, out_path: Path, image_size: int) -> None:
    margin = 24
    gap = 18
    header_h = 78
    label_h = 36
    width = margin * 2 + image_size * len(PANEL_COLUMNS) + gap * (len(PANEL_COLUMNS) - 1)
    height = margin * 2 + header_h + image_size + label_h
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)

    eps = parse_float(row.get("epsilon", ""))
    before = parse_float(row.get("similarity_before", ""))
    after = parse_float(row.get("similarity_after_attack", ""))
    threshold = parse_float(row.get("threshold", ""))
    header = (
        f"{row.get('attack', '')} eps={eps:.3f} "
        f"{row.get('source_name', '')} -> {row.get('target_name', '')}"
    )
    subheader = f"similarity {before:.4f} -> {after:.4f} | threshold {threshold:.4f}"
    draw_text(draw, (margin, margin), header)
    draw_text(draw, (margin, margin + 28), subheader, fill="#333333")

    y = margin + header_h
    for idx, (column, label) in enumerate(PANEL_COLUMNS):
        x = margin + idx * (image_size + gap)
        file_path = package_dir / row[column]
        image = load_image(file_path, image_size)
        panel.paste(image, (x, y))
        draw_text(draw, (x, y + image_size + 8), label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path, quality=95)


def choose_rows(rows: list[dict[str, str]], per_epsilon: int) -> list[dict[str, str]]:
    successful = [row for row in rows if str(row.get("attack_success_before_defense", "")).lower() == "true"]
    by_eps: dict[str, list[dict[str, str]]] = {}
    for row in successful:
        eps = f"{parse_float(row.get('epsilon', '')):.3f}"
        by_eps.setdefault(eps, []).append(row)

    selected: list[dict[str, str]] = []
    for eps in sorted(by_eps):
        candidates = sorted(
            by_eps[eps],
            key=lambda row: parse_float(row.get("similarity_gain", "")),
            reverse=True,
        )
        selected.extend(candidates[:per_epsilon])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Create representative FaceNet verification attack panels.")
    parser.add_argument("--package-dir", type=Path, default=Path("outputs/handoff/facenet_verification_attack_package"))
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/handoff/facenet_verification_attack_panels"))
    parser.add_argument("--per-epsilon", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=180)
    args = parser.parse_args()

    index_path = args.index or args.package_dir / "attack_handoff_index.csv"
    rows = read_rows(index_path)
    selected = choose_rows(rows, args.per_epsilon)
    if not selected:
        raise ValueError(f"No successful rows found in {index_path}")

    for idx, row in enumerate(selected, start=1):
        eps = f"{parse_float(row.get('epsilon', '')):.3f}".replace(".", "")
        out_path = args.out_dir / f"eps{eps}" / f"panel_{idx:02d}_{row['sample_id']}.jpg"
        panel_for_row(row, args.package_dir, out_path, args.image_size)

    print(f"Selected rows: {len(selected)}")
    print(f"Saved panels: {args.out_dir}")


if __name__ == "__main__":
    main()
