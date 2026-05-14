"""Evaluate cosine-similarity face verification from a trained ResNet checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.common.attack_utils import imagenet_normalizer, load_face_model, model_input, pixel_transform


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def make_embedding_fn(model: torch.nn.Module, device: torch.device):
    backbone = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
    backbone.eval()
    to_pixel = pixel_transform()
    mean, std = imagenet_normalizer(device)

    @torch.no_grad()
    def embed(path: str) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        pixel = to_pixel(image).unsqueeze(0).to(device)
        features = backbone(model_input(pixel, mean, std)).flatten(1)
        return F.normalize(features, p=2, dim=1).squeeze(0).cpu()

    return embed


def roc_auc(scores: list[float], labels: list[bool]) -> float:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return float("nan")

    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def metrics_at_threshold(scores: list[float], labels: list[bool], threshold: float) -> dict[str, float]:
    tp = tn = fp = fn = 0
    for score, same in zip(scores, labels):
        accept = score >= threshold
        if same and accept:
            tp += 1
        elif same and not accept:
            fn += 1
        elif not same and accept:
            fp += 1
        else:
            tn += 1

    positives = tp + fn
    negatives = tn + fp
    total = positives + negatives
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / max(total, 1),
        "far": fp / max(negatives, 1),
        "frr": fn / max(positives, 1),
        "tar": tp / max(positives, 1),
        "trr": tn / max(negatives, 1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def choose_threshold(scores: list[float], labels: list[bool], steps: int) -> dict[str, float]:
    lo = min(scores)
    hi = max(scores)
    if lo == hi:
        return metrics_at_threshold(scores, labels, lo)

    best: dict[str, float] | None = None
    for idx in range(steps + 1):
        threshold = lo + (hi - lo) * idx / steps
        current = metrics_at_threshold(scores, labels, threshold)
        current["eer_gap"] = abs(current["far"] - current["frr"])
        current["eer"] = (current["far"] + current["frr"]) / 2
        if best is None or (current["eer_gap"], -current["accuracy"]) < (best["eer_gap"], -best["accuracy"]):
            best = current

    assert best is not None
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate face verification metrics from pair CSV.")
    parser.add_argument("--pairs", type=Path, default=Path("outputs/verification/lfw_test_pairs.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/face_resnet50_lfw10/best.pt"))
    parser.add_argument("--out-scores", type=Path, default=Path("outputs/verification/verification_scores.csv"))
    parser.add_argument("--out-metrics", type=Path, default=Path("outputs/verification/verification_metrics.json"))
    parser.add_argument("--threshold-steps", type=int, default=1000)
    args = parser.parse_args()

    model, classes, device = load_face_model(args.checkpoint)
    embed = make_embedding_fn(model, device)
    embedding_cache: dict[str, torch.Tensor] = {}

    with args.pairs.open(newline="", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    if not pairs:
        raise ValueError(f"No pairs found in {args.pairs}")

    rows: list[dict[str, object]] = []
    scores: list[float] = []
    labels: list[bool] = []

    for row in tqdm(pairs, desc="verification pairs"):
        left_file = row["left_file"]
        right_file = row["right_file"]
        if left_file not in embedding_cache:
            embedding_cache[left_file] = embed(left_file)
        if right_file not in embedding_cache:
            embedding_cache[right_file] = embed(right_file)

        score = float(torch.dot(embedding_cache[left_file], embedding_cache[right_file]))
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
        "checkpoint": str(args.checkpoint),
        "pairs": str(args.pairs),
        "classes": classes,
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
