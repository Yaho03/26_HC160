"""Evaluate clean face verification with a pretrained FaceNet-style model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from tqdm import tqdm

from src.verification.facenet_utils import (
    build_facenet_model,
    cosine_score,
    facenet_embedding,
    facenet_pixel_transform,
    load_facenet_image,
)
from src.verification.evaluate_face_verification import choose_threshold, parse_bool, roc_auc


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FaceNet verification metrics from pair CSV.")
    parser.add_argument("--pairs", type=Path, default=Path("outputs/verification/lfw_test_pairs.csv"))
    parser.add_argument("--pretrained", default="vggface2", choices=["vggface2", "casia-webface"])
    parser.add_argument("--out-scores", type=Path, default=Path("outputs/verification_facenet/verification_scores.csv"))
    parser.add_argument("--out-metrics", type=Path, default=Path("outputs/verification_facenet/verification_metrics.json"))
    parser.add_argument("--threshold-steps", type=int, default=1000)
    args = parser.parse_args()

    model, device = build_facenet_model(args.pretrained)
    to_pixel_tensor = facenet_pixel_transform()
    embedding_cache: dict[str, torch.Tensor] = {}

    with args.pairs.open(newline="", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    if not pairs:
        raise ValueError(f"No pairs found in {args.pairs}")

    rows: list[dict[str, object]] = []
    scores: list[float] = []
    labels: list[bool] = []

    for row in tqdm(pairs, desc="facenet verification pairs"):
        left_file = row["left_file"]
        right_file = row["right_file"]
        if left_file not in embedding_cache:
            left = load_facenet_image(Path(left_file), to_pixel_tensor, device)
            with torch.no_grad():
                embedding_cache[left_file] = facenet_embedding(model, left).squeeze(0).cpu()
        if right_file not in embedding_cache:
            right = load_facenet_image(Path(right_file), to_pixel_tensor, device)
            with torch.no_grad():
                embedding_cache[right_file] = facenet_embedding(model, right).squeeze(0).cpu()

        score = cosine_score(embedding_cache[left_file].unsqueeze(0), embedding_cache[right_file].unsqueeze(0))
        same = parse_bool(row["same_identity"])
        scores.append(score)
        labels.append(same)
        rows.append({
            **row,
            "cosine_similarity": score,
        })

    chosen = choose_threshold(scores, labels, args.threshold_steps)
    auc = roc_auc(scores, labels)
    for row in rows:
        score = float(row["cosine_similarity"])
        same = parse_bool(str(row["same_identity"]))
        accepted = score >= chosen["threshold"]
        row["threshold"] = chosen["threshold"]
        row["accepted"] = accepted
        row["correct"] = accepted == same

    args.out_scores.parent.mkdir(parents=True, exist_ok=True)
    with args.out_scores.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    positives = sum(labels)
    negatives = len(labels) - positives
    metrics = {
        "model": "facenet-pytorch/InceptionResnetV1",
        "pretrained": args.pretrained,
        "pairs": str(args.pairs),
        "samples": len(rows),
        "positive_pairs": positives,
        "negative_pairs": negatives,
        "roc_auc": auc,
        "eer": chosen["eer"],
        "eer_threshold": chosen["threshold"],
        "accuracy_at_eer_threshold": chosen["accuracy"],
        "far_at_eer_threshold": chosen["far"],
        "frr_at_eer_threshold": chosen["frr"],
        "tar_at_eer_threshold": chosen["tar"],
        "trr_at_eer_threshold": chosen["trr"],
        "tp": chosen["tp"],
        "tn": chosen["tn"],
        "fp": chosen["fp"],
        "fn": chosen["fn"],
    }
    args.out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Device: {device}")
    print(f"Model: facenet-pytorch/InceptionResnetV1 pretrained={args.pretrained}")
    print(f"Pairs: {len(rows)} positive={positives} negative={negatives}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"EER: {chosen['eer']:.2%}")
    print(f"Threshold: {chosen['threshold']:.4f}")
    print(f"Accuracy: {chosen['accuracy']:.2%}")
    print(f"FAR: {chosen['far']:.2%}")
    print(f"FRR: {chosen['frr']:.2%}")
    print(f"Scores: {args.out_scores}")
    print(f"Metrics: {args.out_metrics}")


if __name__ == "__main__":
    main()
