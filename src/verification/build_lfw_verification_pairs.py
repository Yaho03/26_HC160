"""Create same/different identity pairs for face verification experiments."""

from __future__ import annotations

import argparse
import csv
import random
from itertools import combinations
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def limited_sample(items: list[tuple[str, str]], limit: int, rng: random.Random) -> list[tuple[str, str]]:
    if limit <= 0 or len(items) <= limit:
        return items
    return rng.sample(items, limit)


def build_positive_pairs(
    by_class: dict[int, list[str]],
    max_per_identity: int,
    rng: random.Random,
) -> list[tuple[str, str, int, int, bool]]:
    rows: list[tuple[str, str, int, int, bool]] = []
    for label, files in sorted(by_class.items()):
        pairs = list(combinations(sorted(files), 2))
        for left, right in limited_sample(pairs, max_per_identity, rng):
            rows.append((left, right, label, label, True))
    return rows


def build_negative_pairs(
    by_class: dict[int, list[str]],
    count: int,
    rng: random.Random,
) -> list[tuple[str, str, int, int, bool]]:
    labels = sorted(label for label, files in by_class.items() if files)
    if len(labels) < 2:
        raise ValueError("Need at least two identities to build negative pairs.")

    rows: list[tuple[str, str, int, int, bool]] = []
    seen: set[tuple[str, str]] = set()
    max_attempts = max(count * 50, 1000)
    attempts = 0

    while len(rows) < count and attempts < max_attempts:
        attempts += 1
        left_label, right_label = rng.sample(labels, 2)
        left = rng.choice(by_class[left_label])
        right = rng.choice(by_class[right_label])
        key = tuple(sorted((left, right)))
        if key in seen:
            continue
        seen.add(key)
        rows.append((left, right, left_label, right_label, False))

    if len(rows) < count:
        raise RuntimeError(f"Only built {len(rows)} negative pairs out of requested {count}.")
    return rows


def load_image_folder(split_dir: Path) -> tuple[list[str], list[tuple[str, int]]]:
    classes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    samples: list[tuple[str, int]] = []
    for label, class_name in enumerate(classes):
        class_dir = split_dir / class_name
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                samples.append((str(path), label))
    return classes, samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LFW verification positive/negative pairs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/lfw_identity_10"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", type=Path, default=Path("outputs/verification/lfw_test_pairs.csv"))
    parser.add_argument("--max-positive-per-identity", type=int, default=30)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_dir = args.data_dir / args.split
    classes, samples = load_image_folder(split_dir)
    if not samples:
        raise FileNotFoundError(f"No images found under {split_dir}")

    by_class: dict[int, list[str]] = {}
    for file, label in samples:
        by_class.setdefault(label, []).append(file)

    rng = random.Random(args.seed)
    positives = build_positive_pairs(by_class, args.max_positive_per_identity, rng)
    negative_count = int(round(len(positives) * args.negative_ratio))
    negatives = build_negative_pairs(by_class, negative_count, rng)

    rows = positives + negatives
    rng.shuffle(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "pair_id",
            "split",
            "left_file",
            "right_file",
            "left_label",
            "right_label",
            "left_name",
            "right_name",
            "same_identity",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (left, right, left_label, right_label, same_identity) in enumerate(rows):
            writer.writerow({
                "pair_id": f"pair_{idx:06d}",
                "split": args.split,
                "left_file": left,
                "right_file": right,
                "left_label": left_label,
                "right_label": right_label,
                "left_name": classes[left_label],
                "right_name": classes[right_label],
                "same_identity": same_identity,
            })

    print(f"Images: {len(samples)}")
    print(f"Identities: {len(classes)}")
    print(f"Positive pairs: {len(positives)}")
    print(f"Negative pairs: {len(negatives)}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
