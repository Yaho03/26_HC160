"""Targeted ZOO-style black-box attack for FaceNet verification.

The attack estimates the gradient of target cosine similarity with symmetric
finite differences. ``queries_used`` counts every verifier evaluation made for
the source or an adversarial candidate and never exceeds ``max_queries``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.common.attack_utils import safe_class_name, save_tensor_image, tensor_norms
from src.verification.evaluate_face_verification import parse_bool
from src.verification.facenet_utils import (
    build_facenet_model,
    cosine_score,
    facenet_embedding,
    facenet_pixel_transform,
    load_facenet_image,
)


def load_threshold(metrics_path: Path | None, threshold: float | None) -> float:
    if threshold is not None:
        return threshold
    if metrics_path is None:
        raise ValueError("Pass --threshold or --metrics with an eer_threshold value.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return float(metrics["eer_threshold"])


def safe_stem(path: Path) -> str:
    return safe_class_name(path.parent.name) + "_" + safe_class_name(path.stem)


def write_rows(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("No attack rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def estimate_coordinate_gradient(
    model: torch.nn.Module,
    adv: torch.Tensor,
    target_emb: torch.Tensor,
    coord_indices: torch.Tensor,
    finite_diff_h: float,
    batch_size: int,
) -> tuple[torch.Tensor, int]:
    """Estimate d(1-cosine)/dx for selected coordinates and return query cost."""
    flat_size = adv.numel()
    gradients = torch.zeros(coord_indices.numel(), device=adv.device)
    base_flat = adv.flatten()
    queries = 0

    for start in range(0, coord_indices.numel(), batch_size):
        coords = coord_indices[start:start + batch_size]
        plus = adv.repeat(coords.numel(), 1, 1, 1)
        minus = adv.repeat(coords.numel(), 1, 1, 1)
        row_indices = torch.arange(coords.numel(), device=adv.device)
        plus_flat = plus.view(coords.numel(), flat_size)
        minus_flat = minus.view(coords.numel(), flat_size)
        plus_flat[row_indices, coords] = (base_flat[coords] + finite_diff_h).clamp(0, 1)
        minus_flat[row_indices, coords] = (base_flat[coords] - finite_diff_h).clamp(0, 1)

        with torch.no_grad():
            plus_emb = facenet_embedding(model, plus)
            minus_emb = facenet_embedding(model, minus)
            expanded_target = target_emb.expand_as(plus_emb)
            plus_loss = 1.0 - torch.nn.functional.cosine_similarity(plus_emb, expanded_target)
            minus_loss = 1.0 - torch.nn.functional.cosine_similarity(minus_emb, expanded_target)
        gradients[start:start + coords.numel()] = (plus_loss - minus_loss) / (2 * finite_diff_h)
        queries += coords.numel() * 2

    return gradients, queries


def validate_args(args: argparse.Namespace) -> None:
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be greater than 0.")
    if args.max_queries < 4:
        raise ValueError("--max-queries must be at least 4.")
    if args.coords_per_iter < 1 or args.fd_batch_size < 1:
        raise ValueError("coordinate and batch sizes must be positive.")
    if args.finite_diff_h <= 0 or args.learning_rate <= 0:
        raise ValueError("finite-difference h and learning rate must be positive.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted ZOO attack for FaceNet verification.")
    parser.add_argument("--pairs", type=Path, default=Path("outputs/verification/lfw_test_pairs.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("outputs/verification_facenet/verification_metrics.json"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--pretrained", default="vggface2", choices=["vggface2", "casia-webface"])
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/verification_attacks_facenet/zoo"))
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--coords-per-iter", type=int, default=32)
    parser.add_argument("--fd-batch-size", type=int, default=32)
    parser.add_argument("--finite-diff-h", type=float, default=0.001)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-positive-pairs", action="store_true")
    parser.add_argument("--only-initial-rejects", action="store_true")
    parser.add_argument("--image-format", default="png", choices=["png", "jpg"])
    args = parser.parse_args()
    validate_args(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    threshold = load_threshold(args.metrics if args.metrics.exists() else None, args.threshold)
    model, device = build_facenet_model(args.pretrained)
    to_pixel_tensor = facenet_pixel_transform()

    with args.pairs.open(newline="", encoding="utf-8") as file:
        pairs = list(csv.DictReader(file))
    if not args.include_positive_pairs:
        pairs = [row for row in pairs if not parse_bool(row["same_identity"])]
    if not pairs:
        raise ValueError(f"No usable pairs found in {args.pairs}")

    image_dir = args.out_dir / "images"
    perturb_dir = args.out_dir / "perturbations"
    image_dir.mkdir(parents=True, exist_ok=True)
    perturb_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    skipped = 0
    for row in tqdm(pairs, desc="targeted facenet verification ZOO"):
        if args.limit > 0 and len(rows) >= args.limit:
            break

        source_path = Path(row["left_file"])
        target_path = Path(row["right_file"])
        source = load_facenet_image(source_path, to_pixel_tensor, device)
        target = load_facenet_image(target_path, to_pixel_tensor, device)
        with torch.no_grad():
            target_emb = facenet_embedding(model, target).detach()
            source_emb = facenet_embedding(model, source)
            similarity_before = cosine_score(source_emb, target_emb)

        before_accept = similarity_before >= threshold
        if args.only_initial_rejects and before_accept:
            skipped += 1
            continue

        start = time.perf_counter()
        adv = source.clone().detach()
        best_similarity = similarity_before
        queries = 1
        iterations = 0
        flat_size = adv.numel()

        while best_similarity < threshold:
            remaining = args.max_queries - queries
            if remaining < 3:
                break
            coord_count = min(args.coords_per_iter, (remaining - 1) // 2, flat_size)
            coord_indices = torch.randperm(flat_size, device=device)[:coord_count]
            gradients, gradient_queries = estimate_coordinate_gradient(
                model,
                adv,
                target_emb,
                coord_indices,
                args.finite_diff_h,
                args.fd_batch_size,
            )
            queries += gradient_queries

            candidate = adv.clone().detach()
            candidate_flat = candidate.flatten()
            source_flat = source.flatten()
            candidate_flat[coord_indices] -= args.learning_rate * gradients.sign()
            delta = (candidate_flat - source_flat).clamp(-args.epsilon, args.epsilon)
            candidate = (source_flat + delta).clamp(0, 1).view_as(source)
            with torch.no_grad():
                candidate_similarity = cosine_score(facenet_embedding(model, candidate), target_emb)
            queries += 1
            iterations += 1
            if candidate_similarity >= best_similarity:
                adv = candidate.detach()
                best_similarity = candidate_similarity

        elapsed = time.perf_counter() - start
        similarity_after = best_similarity
        after_accept = similarity_after >= threshold
        delta = adv - source
        visible_delta = (delta / (2 * args.epsilon)) + 0.5
        l0, l2, linf = tensor_norms(delta)
        suffix = (
            f"{safe_stem(source_path)}_to_{safe_stem(target_path)}"
            f"_facenet_zoo_eps{args.epsilon:.3f}_q{args.max_queries}"
        )
        adv_path = image_dir / f"{suffix}.{args.image_format}"
        perturb_path = perturb_dir / f"{suffix}_perturbation.{args.image_format}"
        save_tensor_image(adv, adv_path)
        save_tensor_image(visible_delta, perturb_path)

        rows.append({
            "pair_id": row["pair_id"],
            "source_file": str(source_path),
            "target_enroll_file": str(target_path),
            "adv_file": str(adv_path),
            "perturbation_file": str(perturb_path),
            "attack": "targeted_zoo_facenet_verification",
            "model": "facenet-pytorch/InceptionResnetV1",
            "pretrained": args.pretrained,
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
            "attack_success": after_accept,
            "success_from_reject": (not before_accept) and after_accept,
            "epsilon": args.epsilon,
            "alpha": args.learning_rate,
            "steps": iterations,
            "max_queries": args.max_queries,
            "queries_used": queries,
            "query_budget_exhausted": queries >= args.max_queries,
            "coords_per_iter": args.coords_per_iter,
            "finite_diff_h": args.finite_diff_h,
            "seed": args.seed,
            "only_initial_rejects": args.only_initial_rejects,
            "l0": l0,
            "l2": l2,
            "linf": linf,
            "time_sec": elapsed,
        })

    metadata_path = args.out_dir / (
        f"metadata_targeted_zoo_facenet_verification_eps{args.epsilon:.3f}"
        f"_queries{args.max_queries}.csv"
    )
    write_rows(rows, metadata_path)
    n = len(rows)
    print(f"Device: {device}")
    print(f"Pairs: {n}  |  Skipped: {skipped}")
    print(f"Threshold: {threshold:.4f}")
    print(f"ASR: {sum(bool(row['attack_success']) for row in rows) / n:.2%}")
    print(f"Avg queries used: {sum(int(row['queries_used']) for row in rows) / n:.1f}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
