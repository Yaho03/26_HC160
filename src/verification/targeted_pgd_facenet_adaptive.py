"""Adaptive targeted PGD: attacks through a defense transform so the perturbation survives it.

핵심 아이디어:
  일반 PGD는 clean 텐서에서 gradient를 구하기 때문에, smoothing 같은 전처리 방어를 적용하면
  perturbation이 희석되어 공격이 무력화됨 (방어팀 결과: smoothing 78.8% 방어 성공).

  Adaptive PGD는 PGD 루프 안에서 defense transform을 먼저 적용하고 embedding을 계산하여
  gradient를 방어를 "통과"하도록 만든다. 그 결과 perturbation이 smoothing 후에도 살아남도록
  최적화된다.

지원 방어 transform:
  smoothing  — Gaussian blur (방어팀 radius=3 기준 kernel_size=13, sigma=3.0)
  jpeg       — Differentiable JPEG 근사 (픽셀 반올림으로 근사; gradient는 straight-through 사용)
  none       — 일반 PGD (기준 비교용)
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

from src.common.attack_utils import safe_class_name, save_tensor_image, tensor_norms
from src.verification.evaluate_face_verification import parse_bool
from src.verification.facenet_utils import (
    build_facenet_model,
    cosine_score,
    facenet_embedding,
    facenet_pixel_transform,
    load_facenet_image,
)


# ------------------------------------------------------------------
# Defense transforms (differentiable)
# ------------------------------------------------------------------

def _gaussian_kernel(kernel_size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    """Gaussian kernel for F.conv2d (differentiable)."""
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()


def apply_smoothing(x: torch.Tensor, kernel_size: int = 13, sigma: float = 3.0) -> torch.Tensor:
    """Gaussian blur applied channel-wise. kernel_size=13, sigma=3.0 ≈ PIL GaussianBlur radius=3."""
    channels = x.shape[1]
    kernel = _gaussian_kernel(kernel_size, sigma, channels, x.device)
    padding = kernel_size // 2
    return F.conv2d(x, kernel, padding=padding, groups=channels)


def apply_jpeg_approx(x: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """JPEG 근사: 픽셀값을 양자화 경계로 반올림.
    실제 JPEG DCT 변환은 비미분이라 straight-through estimator를 쓴다.
    (forward: round, backward: identity)
    quality=75 → 약 2픽셀 단위 양자화 근사.
    """
    step = (100 - quality) / 100.0 * 0.08 + 0.01  # 경험적 추정 (질 낮을수록 step 큼)
    x_q = torch.round(x / step) * step
    # straight-through: backward에서 x_q 대신 x의 gradient 사용
    return x + (x_q - x).detach()


def load_dae_transform(checkpoint_path: str, device: torch.device, base_ch: int = 32):
    """학습된 DAE를 방어 transform으로 로드.
    BPDA (Backward Pass Differentiable Approximation): forward는 DAE 통과,
    backward는 DAE가 미분 가능하므로 실제 gradient가 흐름.
    """
    from src.defenses.dae_model import load_dae
    dae = load_dae(checkpoint_path, device, base_ch)
    dae.eval()

    def fn(x: torch.Tensor) -> torch.Tensor:
        return dae(x)

    return fn


def make_transform(
    name: str,
    smoothing_kernel: int,
    smoothing_sigma: float,
    jpeg_quality: int,
    dae_checkpoint: str | None = None,
    dae_base_ch: int = 32,
    device: torch.device | None = None,
):
    if name == "smoothing":
        def fn(x: torch.Tensor) -> torch.Tensor:
            return apply_smoothing(x, smoothing_kernel, smoothing_sigma)
    elif name == "jpeg":
        def fn(x: torch.Tensor) -> torch.Tensor:
            return apply_jpeg_approx(x, jpeg_quality)
    elif name == "dae":
        if not dae_checkpoint:
            raise ValueError("--dae-checkpoint 경로를 지정하세요 (--defense-transform dae 사용 시).")
        fn = load_dae_transform(dae_checkpoint, device, dae_base_ch)
    else:
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x
    return fn


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive targeted PGD for FaceNet verification — attacks through a defense transform."
    )
    parser.add_argument("--pairs", type=Path, default=Path("outputs/verification/lfw_test_pairs.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("outputs/verification_facenet/verification_metrics.json"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--pretrained", default="vggface2", choices=["vggface2", "casia-webface"])
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/verification_attacks_facenet/pgd_adaptive"))
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--steps", type=int, default=20,
                        help="Adaptive attack는 일반 PGD보다 step을 더 늘리는 게 좋다 (기본 20).")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--include-positive-pairs", action="store_true")
    parser.add_argument("--only-initial-rejects", action="store_true")
    parser.add_argument("--image-format", default="png", choices=["png", "jpg"])
    # Defense transform to attack through
    parser.add_argument(
        "--defense-transform", default="smoothing",
        choices=["smoothing", "jpeg", "dae", "none"],
        help="Attack through this defense transform (default: smoothing).",
    )
    parser.add_argument("--smoothing-kernel", type=int, default=13,
                        help="Gaussian blur kernel size (must be odd). 13 ≈ PIL GaussianBlur radius=3.")
    parser.add_argument("--smoothing-sigma", type=float, default=3.0,
                        help="Gaussian blur sigma. 3.0 ≈ PIL GaussianBlur radius=3.")
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--dae-checkpoint", type=str, default=None,
                        help="DAE 체크포인트 경로 (--defense-transform dae 사용 시 필수).")
    parser.add_argument("--dae-base-ch", type=int, default=32)
    args = parser.parse_args()

    threshold = load_threshold(args.metrics if args.metrics.exists() else None, args.threshold)
    model, device = build_facenet_model(args.pretrained)
    to_pixel_tensor = facenet_pixel_transform()
    defense_fn = make_transform(
        args.defense_transform,
        args.smoothing_kernel,
        args.smoothing_sigma,
        args.jpeg_quality,
        dae_checkpoint=args.dae_checkpoint,
        dae_base_ch=args.dae_base_ch,
        device=device,
    )

    with args.pairs.open(newline="", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    if not args.include_positive_pairs:
        pairs = [row for row in pairs if not parse_bool(row["same_identity"])]
    if not pairs:
        raise ValueError(f"No usable pairs found in {args.pairs}")

    attack_name = f"targeted_pgd_facenet_adaptive_{args.defense_transform}"
    image_dir = args.out_dir / "images"
    perturb_dir = args.out_dir / "perturbations"
    image_dir.mkdir(parents=True, exist_ok=True)
    perturb_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    skipped_initial_accepts = 0

    for row in tqdm(pairs, desc=f"adaptive PGD ({args.defense_transform})"):
        if args.limit > 0 and len(rows) >= args.limit:
            break

        source_path = Path(row["left_file"])
        target_path = Path(row["right_file"])
        source = load_facenet_image(source_path, to_pixel_tensor, device)
        target = load_facenet_image(target_path, to_pixel_tensor, device)

        start = time.perf_counter()
        with torch.no_grad():
            source_emb = facenet_embedding(model, source)
            target_emb = facenet_embedding(model, target).detach()
            similarity_before = cosine_score(source_emb, target_emb)

        before_accept = similarity_before >= threshold
        if args.only_initial_rejects and before_accept:
            skipped_initial_accepts += 1
            continue

        adv = source.clone().detach()
        for _ in range(args.steps):
            adv = adv.clone().detach().requires_grad_(True)
            # Defense transform 통과 후 embedding 계산 — gradient가 방어를 통과해서 흐름
            adv_defended = defense_fn(adv)
            adv_emb = facenet_embedding(model, adv_defended)
            loss = 1.0 - F.cosine_similarity(adv_emb, target_emb).mean()
            model.zero_grad(set_to_none=True)
            loss.backward()
            adv = adv - args.alpha * adv.grad.sign()
            delta = torch.clamp(adv - source, min=-args.epsilon, max=args.epsilon)
            adv = (source + delta).detach().clamp(0, 1)

        # 평가: 방어 후 similarity (공격이 방어를 버텼는지)
        with torch.no_grad():
            adv_emb_clean = facenet_embedding(model, adv)
            adv_emb_defended = facenet_embedding(model, defense_fn(adv))
            similarity_after_clean = cosine_score(adv_emb_clean, target_emb)
            similarity_after_defended = cosine_score(adv_emb_defended, target_emb)

        elapsed = time.perf_counter() - start

        delta = adv - source
        visible_delta = (delta / (2 * args.epsilon)) + 0.5
        l0, l2, linf = tensor_norms(delta)

        after_accept_clean = similarity_after_clean >= threshold
        after_accept_defended = similarity_after_defended >= threshold
        attack_success = after_accept_clean
        # adaptive attack의 핵심 지표: 방어 후에도 accept 유지되는지
        adaptive_success = after_accept_defended
        success_from_reject = (not before_accept) and after_accept_clean

        suffix = (
            f"{safe_stem(source_path)}_to_{safe_stem(target_path)}"
            f"_facenet_adaptive_{args.defense_transform}_eps{args.epsilon:.3f}"
            f"_s{args.steps}"
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
            "attack": attack_name,
            "model": "facenet-pytorch/InceptionResnetV1",
            "pretrained": args.pretrained,
            "source_label": row["left_label"],
            "target_label": row["right_label"],
            "source_name": row["left_name"],
            "target_name": row["right_name"],
            "same_identity_pair": row["same_identity"],
            "threshold": threshold,
            "similarity_before": similarity_before,
            "similarity_after": similarity_after_clean,
            "similarity_after_defended": similarity_after_defended,
            "similarity_gain": similarity_after_clean - similarity_before,
            "similarity_gain_defended": similarity_after_defended - similarity_before,
            "accepted_before": before_accept,
            "accepted_after": after_accept_clean,
            "accepted_after_defended": after_accept_defended,
            "attack_success": attack_success,
            "adaptive_success": adaptive_success,
            "success_from_reject": success_from_reject,
            "defense_transform": args.defense_transform,
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "steps": args.steps,
            "only_initial_rejects": args.only_initial_rejects,
            "l0": l0,
            "l2": l2,
            "linf": linf,
            "time_sec": elapsed,
        })

    metadata_path = args.out_dir / (
        f"metadata_{attack_name}_eps{args.epsilon:.3f}_s{args.steps}.csv"
    )
    write_rows(rows, metadata_path)

    n = len(rows)
    asr = sum(bool(r["attack_success"]) for r in rows) / n
    adaptive_asr = sum(bool(r["adaptive_success"]) for r in rows) / n
    avg_gain = sum(float(r["similarity_gain"]) for r in rows) / n
    avg_gain_def = sum(float(r["similarity_gain_defended"]) for r in rows) / n
    avg_l2 = sum(float(r["l2"]) for r in rows) / n

    print(f"Device: {device}")
    print(f"Model: facenet-pytorch/InceptionResnetV1 pretrained={args.pretrained}")
    print(f"Defense transform attacked through: {args.defense_transform}")
    print(f"Pairs: {n}  |  Skipped initial accepts: {skipped_initial_accepts}")
    print(f"Threshold: {threshold:.4f}")
    print(f"ASR (clean eval):    {asr:.2%}  — adv 이미지 자체의 공격 성공률")
    print(f"ASR (defended eval): {adaptive_asr:.2%}  — 방어 적용 후에도 accept 유지되는 비율")
    print(f"Avg similarity gain (clean):    {avg_gain:.4f}")
    print(f"Avg similarity gain (defended): {avg_gain_def:.4f}")
    print(f"Avg L2: {avg_l2:.4f}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
