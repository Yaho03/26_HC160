"""Targeted PGD impersonation attack for face verification.

This attack changes a source face image so its embedding becomes closer to a
target enrollment image. Success means the adversarial source image crosses the
verification threshold for the target identity.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.common.attack_utils import (
    imagenet_normalizer,
    load_clean_image,
    load_face_model,
    model_input,
    pixel_transform,
    safe_class_name,
    save_tensor_image,
    tensor_norms,
)


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def load_threshold(metrics_path: Path | None, threshold: float | None) -> float:
    if threshold is not None:
        return threshold
    if metrics_path is None:
        raise ValueError("Pass --threshold or --metrics with an eer_threshold value.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "eer_threshold" not in metrics:
        raise KeyError(f"{metrics_path} does not contain eer_threshold")
    return float(metrics["eer_threshold"])


def build_embedding_backbone(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    backbone = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
    backbone.eval()
    return backbone


def embedding(
    backbone: torch.nn.Module,
    pixel_tensor: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    features = backbone(model_input(pixel_tensor, mean, std)).flatten(1)
    return F.normalize(features, p=2, dim=1)


def cosine_score(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left, right).detach().cpu().item())


def safe_stem(path: Path) -> str:
    return safe_class_name(path.parent.name) + "_" + safe_class_name(path.stem)


def write_rows(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("No attack rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted PGD attack for face verification.")
    parser.add_argument("--pairs", type=Path, default=Path("outputs/verification/lfw_test_pairs.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/face_resnet50_lfw10/best.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("outputs/verification/verification_metrics.json"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/verification_attacks/pgd"))
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=0.003)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--include-positive-pairs", action="store_true")
    parser.add_argument("--random-start", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    threshold = load_threshold(args.metrics if args.metrics.exists() else None, args.threshold)
    model, _classes, device = load_face_model(args.checkpoint)
    backbone = build_embedding_backbone(model, device)
    mean, std = imagenet_normalizer(device)
    to_pixel_tensor = pixel_transform()

    with args.pairs.open(newline="", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    if not args.include_positive_pairs:
        pairs = [row for row in pairs if not parse_bool(row["same_identity"])]
    if args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        raise ValueError(f"No usable pairs found in {args.pairs}")

    image_dir = args.out_dir / "images"
    perturb_dir = args.out_dir / "perturbations"
    image_dir.mkdir(parents=True, exist_ok=True)
    perturb_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for row in tqdm(pairs, desc="targeted verification PGD"):
        source_path = Path(row["left_file"])
        target_path = Path(row["right_file"])
        source = load_clean_image(source_path, to_pixel_tensor, device)
        target = load_clean_image(target_path, to_pixel_tensor, device)

        start = time.perf_counter()
        with torch.no_grad():
            source_emb = embedding(backbone, source, mean, std)
            target_emb = embedding(backbone, target, mean, std).detach()
            similarity_before = cosine_score(source_emb, target_emb)

        if args.random_start:
            adv = (source + torch.empty_like(source).uniform_(-args.epsilon, args.epsilon)).clamp(0, 1)
        else:
            adv = source.clone().detach()

        for _ in range(args.steps):
            adv = adv.clone().detach().requires_grad_(True)
            adv_emb = embedding(backbone, adv, mean, std)
            similarity = F.cosine_similarity(adv_emb, target_emb).mean()
            loss = 1.0 - similarity
            backbone.zero_grad(set_to_none=True)
            loss.backward()
            adv = adv - args.alpha * adv.grad.sign()
            delta = torch.clamp(adv - source, min=-args.epsilon, max=args.epsilon)
            adv = (source + delta).detach().clamp(0, 1)

        with torch.no_grad():
            adv_emb = embedding(backbone, adv, mean, std)
            similarity_after = cosine_score(adv_emb, target_emb)
        elapsed = time.perf_counter() - start

        delta = adv - source
        visible_delta = (delta / (2 * args.epsilon)) + 0.5
        l0, l2, linf = tensor_norms(delta)

        before_accept = similarity_before >= threshold
        after_accept = similarity_after >= threshold
        attack_success = after_accept
        success_from_reject = (not before_accept) and after_accept

        suffix = (
            f"{safe_stem(source_path)}_to_{safe_stem(target_path)}"
            f"_eps{args.epsilon:.3f}_a{args.alpha:.3f}_s{args.steps}"
        )
        adv_path = image_dir / f"{suffix}.jpg"
        perturb_path = perturb_dir / f"{suffix}_perturbation.jpg"
        save_tensor_image(adv, adv_path)
        save_tensor_image(visible_delta, perturb_path)

        rows.append({
            "pair_id": row["pair_id"],
            "source_file": str(source_path),
            "target_enroll_file": str(target_path),
            "adv_file": str(adv_path),
            "perturbation_file": str(perturb_path),
            "attack": "targeted_pgd_verification",
            "source_label": row["left_label"],
            "target_label": row["right_label"],
            "source_name": row["left_name"],
            "target_name": row["right_name"],
            "same_identity_pair": row["same_identity"],
            "threshold": threshold,
            "similarity_before": similarity_before,
            "similarity_after": similarity_after,
            "similarity_gain": similarity_after - similarity_before,
            "accepted_before": before_accept,
            "accepted_after": after_accept,
            "attack_success": attack_success,
            "success_from_reject": success_from_reject,
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "steps": args.steps,
            "random_start": args.random_start,
            "l0": l0,
            "l2": l2,
            "linf": linf,
            "time_sec": elapsed,
        })

    metadata_path = args.out_dir / (
        f"metadata_targeted_pgd_verification_eps{args.epsilon:.3f}"
        f"_alpha{args.alpha:.3f}_steps{args.steps}.csv"
    )
    write_rows(rows, metadata_path)

    attack_success_rate = sum(bool(row["attack_success"]) for row in rows) / len(rows)
    success_from_reject_rate = sum(bool(row["success_from_reject"]) for row in rows) / len(rows)
    avg_gain = sum(float(row["similarity_gain"]) for row in rows) / len(rows)
    avg_l2 = sum(float(row["l2"]) for row in rows) / len(rows)
    avg_linf = sum(float(row["linf"]) for row in rows) / len(rows)

    print(f"Device: {device}")
    print(f"Pairs: {len(rows)}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Target accept rate after attack: {attack_success_rate:.2%}")
    print(f"Success from reject rate: {success_from_reject_rate:.2%}")
    print(f"Avg similarity gain: {avg_gain:.4f}")
    print(f"Avg L2: {avg_l2:.4f}")
    print(f"Avg Linf: {avg_linf:.4f}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
