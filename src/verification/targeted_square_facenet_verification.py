"""Targeted Square Attack (black-box) for FaceNet verification.

블랙박스 공격: gradient 없이 forward pass(cosine similarity)만 사용.
random square 패치를 시도해서 target embedding과의 cosine similarity를 높인다.

classification Square와의 차이:
  - score: cosine_similarity(embedding(adv), embedding(target)) 높을수록 좋음
  - "success": similarity >= threshold
  - gradient 불필요 → white-box 모델 정보 없어도 동작

현실 시나리오 적용:
  금융 얼굴인증 API에 요청을 보낼 수 있지만 모델 내부는 모를 때 사용.
  쿼리 수(max_queries)가 공격 비용 지표.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
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
    import json
    if threshold is not None:
        return threshold
    if metrics_path is None:
        raise ValueError("Pass --threshold or --metrics.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return float(metrics["eer_threshold"])


def safe_stem(path: Path) -> str:
    return safe_class_name(path.parent.name) + "_" + safe_class_name(path.stem)


def square_size(iteration: int, max_queries: int, image_size: int, p_init: float) -> int:
    progress = iteration / max(max_queries, 1)
    p = p_init * (1.0 - progress) + 0.01 * progress
    area = max(1, int(p * image_size * image_size))
    return max(1, min(image_size, int(math.sqrt(area))))


def write_rows(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Targeted Square Attack (black-box) for FaceNet verification."
    )
    parser.add_argument("--pairs", type=Path, default=Path("outputs/verification/lfw_test_pairs.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("outputs/verification_facenet/verification_metrics.json"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--pretrained", default="vggface2", choices=["vggface2", "casia-webface"])
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/verification_attacks_facenet/square"))
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--p-init", type=float, default=0.05,
                        help="초기 square 면적 비율. 0.05=이미지의 5%.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-positive-pairs", action="store_true")
    parser.add_argument("--only-initial-rejects", action="store_true")
    parser.add_argument("--image-format", default="png", choices=["png", "jpg"])
    args = parser.parse_args()

    if args.epsilon <= 0:
        raise ValueError("--epsilon must be greater than 0.")
    if args.max_queries < 2:
        raise ValueError("--max-queries must be at least 2.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    threshold = load_threshold(args.metrics if args.metrics.exists() else None, args.threshold)
    model, device = build_facenet_model(args.pretrained)
    to_pixel_tensor = facenet_pixel_transform()

    with args.pairs.open(newline="", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    if not args.include_positive_pairs:
        pairs = [row for row in pairs if not parse_bool(row["same_identity"])]
    if not pairs:
        raise ValueError(f"No usable pairs in {args.pairs}")

    image_dir = args.out_dir / "images"
    perturb_dir = args.out_dir / "perturbations"
    image_dir.mkdir(parents=True, exist_ok=True)
    perturb_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    skipped = 0

    for row in tqdm(pairs, desc="targeted Square verification"):
        if args.limit > 0 and len(rows) >= args.limit:
            break

        source_path = Path(row["left_file"])
        target_path = Path(row["right_file"])
        source = load_facenet_image(source_path, to_pixel_tensor, device)
        target = load_facenet_image(target_path, to_pixel_tensor, device)

        with torch.no_grad():
            target_emb = facenet_embedding(model, target)
            source_emb = facenet_embedding(model, source)
            similarity_before = cosine_score(source_emb, target_emb)

        before_accept = similarity_before >= threshold
        if args.only_initial_rejects and before_accept:
            skipped += 1
            continue

        _, _, h, w = source.shape
        start = time.perf_counter()

        # 초기 random perturbation
        delta = torch.empty_like(source).uniform_(-args.epsilon, args.epsilon)
        adv = (source + delta).clamp(0, 1)
        with torch.no_grad():
            adv_emb = facenet_embedding(model, adv)
            best_sim = cosine_score(adv_emb, target_emb)

        # source 기준 평가와 초기 무작위 후보 평가를 쿼리 예산에 포함한다.
        queries = 2

        for query_idx in range(1, args.max_queries):
            if best_sim >= threshold or queries >= args.max_queries:
                break

            size = square_size(query_idx, args.max_queries, h, args.p_init)
            top = random.randint(0, h - size)
            left = random.randint(0, w - size)

            candidate_delta = delta.clone()
            patch = torch.empty((1, 3, size, size), device=device).uniform_(-args.epsilon, args.epsilon)
            candidate_delta[:, :, top:top + size, left:left + size] = patch
            candidate_delta.clamp_(-args.epsilon, args.epsilon)
            candidate = (source + candidate_delta).clamp(0, 1)

            with torch.no_grad():
                cand_emb = facenet_embedding(model, candidate)
                cand_sim = cosine_score(cand_emb, target_emb)
            queries += 1

            if cand_sim >= best_sim:
                delta = candidate_delta
                adv = candidate
                best_sim = cand_sim

        similarity_after = best_sim
        elapsed = time.perf_counter() - start

        after_accept = similarity_after >= threshold
        final_delta = adv - source
        visible_delta = (final_delta / (2 * args.epsilon)) + 0.5
        l0, l2, linf = tensor_norms(final_delta)

        suffix = (
            f"{safe_stem(source_path)}_to_{safe_stem(target_path)}"
            f"_facenet_square_eps{args.epsilon:.3f}_q{args.max_queries}"
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
            "attack": "targeted_square_facenet_verification",
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
            "alpha": 0.0,
            "steps": queries,
            "max_queries": args.max_queries,
            "queries_used": queries,
            "p_init": args.p_init,
            "seed": args.seed,
            "only_initial_rejects": args.only_initial_rejects,
            "l0": l0,
            "l2": l2,
            "linf": linf,
            "time_sec": elapsed,
        })

    metadata_path = args.out_dir / (
        f"metadata_targeted_square_facenet_verification"
        f"_eps{args.epsilon:.3f}_q{args.max_queries}.csv"
    )
    write_rows(rows, metadata_path)

    n = len(rows)
    asr = sum(bool(r["attack_success"]) for r in rows) / n
    avg_sim_gain = sum(float(r["similarity_gain"]) for r in rows) / n
    avg_queries = sum(int(r["queries_used"]) for r in rows) / n
    avg_l2 = sum(float(r["l2"]) for r in rows) / n

    print(f"Device: {device}")
    print(f"Model: facenet-pytorch/InceptionResnetV1 pretrained={args.pretrained}")
    print(f"Pairs: {n}  |  Skipped: {skipped}")
    print(f"Threshold: {threshold:.4f}")
    print(f"ASR: {asr:.2%}")
    print(f"Avg similarity gain: {avg_sim_gain:.4f}")
    print(f"Avg queries used: {avg_queries:.1f} / {args.max_queries}")
    print(f"Avg L2: {avg_l2:.4f}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
