"""
Adaptive attack — 공격자가 방어를 알고 있는 경우

07_DEFENSE_AND_DETECTION_SPEC.md 3절과 7절은 defended 대상 adaptive attack 평가를
요구한다. 지금까지의 결과는 공격자가 squeezing detector의 존재를 모른다는 가정에
기댄다. 그 가정이 깨지면 탐지율이 어떻게 되는지가 실제 보안 주장의 근거다.

Squeezing detector는 변환 전후 임베딩 차이를 본다. 따라서 공격자의 목표는 둘이다.

1. 등록자와의 유사도를 높인다 (기존 공격과 동일)
2. 변환 전후 임베딩 차이를 작게 유지한다 (탐지 회피)

EOT는 변환을 공격 루프 안에 넣어 두 목표를 함께 최적화한다. 변환은 미분 가능해야
하므로 blur와 저해상도만 쓴다. JPEG와 median filter는 미분 불가라 EOT 루프에 넣을
수 없다. 이는 방어 쪽에 유리한 조건이며 결과 해석에 반드시 명시한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from src.verification.defenses.facenet_embed import get_model, preprocess

# 미분 가능한 근사 변환. 이름은 squeeze_probe의 변환과 대응한다.
_DIFFERENTIABLE = {
    "blur0.5": {"kind": "blur", "sigma": 0.5},
    "blur0.8": {"kind": "blur", "sigma": 0.8},
    "blur1.2": {"kind": "blur", "sigma": 1.2},
    "blur2.0": {"kind": "blur", "sigma": 2.0},
    "lowres64": {"kind": "lowres", "size": 64},
    "lowres32": {"kind": "lowres", "size": 32},
}

MODES = {
    "oblivious": {"uses_defense": False, "note": "방어를 모르는 표준 PGD"},
    "eot": {"uses_defense": True, "note": "미분 가능한 변환만 루프에 넣어 탐지를 회피"},
    "bpda": {
        "uses_defense": True,
        "note": "미분 불가 변환의 backward를 항등으로 근사해 전부 최적화",
    },
}


class UnknownAdaptiveModeError(ValueError):
    """선언되지 않은 공격 모드."""


@dataclass(frozen=True)
class AdaptiveAttackConfig:
    epsilon: float = 0.03
    steps: int = 40
    step_size: float = 0.002
    # 탐지 회피에 얼마나 비중을 둘지. 0이면 기존 공격과 같다.
    consistency_weight: float = 1.0


def resolve_mode(mode: str) -> dict:
    if mode not in MODES:
        raise UnknownAdaptiveModeError(
            f"알 수 없는 모드 {mode!r}. 사용 가능: {sorted(MODES)}"
        )
    return MODES[mode]


def build_eot_transforms(names) -> list[str]:
    """
    EOT 루프에 넣을 변환을 고른다. 미분 불가한 변환은 여기 들어올 수 없다.
    """
    names = list(names)
    if not names:
        raise ValueError("EOT 변환을 하나 이상 지정해야 한다")
    usable = [n for n in names if n in _DIFFERENTIABLE]
    if not usable:
        raise KeyError(
            f"미분 가능한 변환이 없다: {names}. 사용 가능: {sorted(_DIFFERENTIABLE)}"
        )
    return usable


def _gaussian_kernel(sigma: float, device, dtype):
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return (kernel / kernel.sum()), radius


def _apply(tensor: torch.Tensor, spec: dict) -> torch.Tensor:
    """미분 가능한 변환. 입력은 (1,3,H,W)."""
    if spec["kind"] == "blur":
        kernel, radius = _gaussian_kernel(spec["sigma"], tensor.device, tensor.dtype)
        channels = tensor.shape[1]
        horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
        vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
        padded = F.pad(tensor, (radius, radius, 0, 0), mode="reflect")
        blurred = F.conv2d(padded, horizontal, groups=channels)
        padded = F.pad(blurred, (0, 0, radius, radius), mode="reflect")
        return F.conv2d(padded, vertical, groups=channels)
    if spec["kind"] == "lowres":
        size = spec["size"]
        small = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
        return F.interpolate(
            small, size=tensor.shape[-2:], mode="bilinear", align_corners=False
        )
    raise KeyError(spec["kind"])


def run_eot_attack(
    source: Image.Image,
    target_embedding: torch.Tensor,
    config: AdaptiveAttackConfig,
    *,
    transforms,
    use_bpda: bool = False,
    device=None,
) -> tuple[Image.Image, float]:
    """
    등록자 유사도를 높이면서 변환 전후 임베딩 차이를 작게 유지한다.

    두 번째 항이 탐지 회피다. consistency_weight가 0이면 기존 PGD와 같다.

    use_bpda가 참이면 미분 불가한 변환도 루프에 넣는다. backward만 항등으로
    근사하므로 공격자가 보는 detector 출력은 방어가 실제로 계산하는 값과 같다.
    """
    if use_bpda:
        from src.verification.defenses.bpda import bpda_transform

        names = list(transforms)
        specs = None
    else:
        names = build_eot_transforms(transforms)
        specs = [_DIFFERENTIABLE[name] for name in names]

    model, dev = get_model(device)
    target = target_embedding.to(dev).unsqueeze(0)

    original = preprocess(source).to(dev)
    adversarial = original.clone()

    for _ in range(config.steps):
        adversarial.requires_grad_(True)
        embedding = F.normalize(model(adversarial), p=2, dim=1)

        identity_loss = -F.cosine_similarity(embedding, target).mean()

        consistency_loss = torch.zeros((), device=dev)
        if config.consistency_weight > 0:
            variants = (
                [bpda_transform(adversarial, name) for name in names]
                if use_bpda
                else [_apply(adversarial, spec) for spec in specs]
            )
            for variant in variants:
                squeezed = F.normalize(model(variant), p=2, dim=1)
                # detector가 보는 값. 작을수록 탐지되지 않는다.
                consistency_loss = consistency_loss + (
                    1.0 - F.cosine_similarity(embedding, squeezed).mean()
                )
            consistency_loss = consistency_loss / len(variants)

        loss = identity_loss + config.consistency_weight * consistency_loss
        gradient = torch.autograd.grad(loss, adversarial)[0]

        with torch.no_grad():
            adversarial = adversarial - config.step_size * gradient.sign()
            delta = torch.clamp(adversarial - original, -config.epsilon, config.epsilon)
            adversarial = (original + delta).detach()

    with torch.no_grad():
        final = F.normalize(model(adversarial), p=2, dim=1)
        similarity = float(F.cosine_similarity(final, target).item())

    array = adversarial.squeeze(0).permute(1, 2, 0).cpu().numpy()
    array = (array * 128.0 + 127.5).clip(0, 255).astype("uint8")
    return Image.fromarray(array), similarity


# ── 평가 CLI ──────────────────────────────────────────────────────────────────


def detector_score(crop, enroll_vector, embedder, features, statistics) -> float:
    """
    계측 세션에서 정한 특징과 clean 통계로 detector 점수를 재현한다.

    정규화 통계는 clean 표본에서만 나온 값이다. 공격 표본으로 다시 맞추면
    07_DEFENSE_AND_DETECTION_SPEC.md 5절을 어긴다.
    """
    from src.verification.defenses.squeeze_probe import probe_crop

    reading = probe_crop(crop, enroll_vector, embedder)
    by_transform = {item.transform: item for item in reading.readings}
    total = 0.0
    for transform, measure in features:
        item = by_transform[transform]
        value = (
            1.0 - item.cos_orig_transformed
            if measure == "self_consistency"
            else abs(item.cos_orig_enroll - item.cos_transformed_enroll)
        )
        mean, deviation = statistics[(transform, measure)]
        total += (value - mean) / deviation
    return total


def calibration_from_probe(probe_csv, top_k: int = 6, target_fpr: float = 0.01):
    """계측 세션에서 특징 선택, clean 통계, 임계값을 가져온다."""
    import numpy as np

    from src.verification.defenses.probe_analyze import (
        combine_clean_normalized,
        feature_table,
        load_probe_rows,
        roc_auc,
        threshold_at_fpr,
    )

    table = feature_table(load_probe_rows(probe_csv))
    ranked = sorted(
        (
            key
            for key in table
            if roc_auc(table[key]["adversarial"], table[key]["clean"]) is not None
        ),
        key=lambda key: -roc_auc(table[key]["adversarial"], table[key]["clean"]),
    )[:top_k]

    statistics = {}
    for key in ranked:
        values = np.asarray(table[key]["clean"], dtype=float)
        deviation = values.std()
        statistics[key] = (values.mean(), deviation if deviation > 1e-12 else 1.0)

    clean_scores, _ = combine_clean_normalized(table, ranked)
    return ranked, statistics, threshold_at_fpr(clean_scores, target_fpr)


def main() -> int:
    import argparse
    import csv
    import json

    import numpy as np
    from PIL import Image

    from src.verification.defenses.facenet_embed import (
        FaceNetBatchEmbedder,
        get_embedding,
    )
    from src.verification.defenses.verification_defense_temporal_camera import (
        generate_adversarial,
    )

    parser = argparse.ArgumentParser(
        description="방어를 아는 공격과 모르는 공격의 탐지율 비교"
    )
    parser.add_argument("--package", required=True, help="공격 패키지 루트")
    parser.add_argument("--probe", required=True, help="임계값을 정한 계측 CSV")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--identity-threshold", type=float, required=True)
    parser.add_argument(
        "--weights", default="0,1,5", help="쉼표로 구분한 consistency_weight 목록"
    )
    parser.add_argument(
        "--bpda",
        action="store_true",
        help="미분 불가 변환도 공격 대상에 넣는다. detector가 쓰는 변환을 그대로 쓴다",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.package)
    features, statistics, threshold = calibration_from_probe(
        args.probe, target_fpr=args.target_fpr
    )
    # BPDA면 detector가 실제로 쓰는 변환을 그대로 공격 대상에 넣는다.
    attack_transforms = (
        sorted({transform for transform, _ in features})
        if args.bpda
        else ["blur0.5", "blur0.8", "blur1.2"]
    )
    print(f"detector 특징 {[f'{a}|{b}' for a, b in features]}")
    print(f"공격 대상 변환 {attack_transforms}")
    print(f"임계값 {threshold:.4f}  (목표 FPR {args.target_fpr})\n")

    weights = [float(w) for w in args.weights.split(",") if w.strip()]
    embedder = FaceNetBatchEmbedder()
    pairs = list(csv.DictReader((root / "attack_handoff_index.csv").open()))[: args.limit]

    prefix = "bpda" if args.bpda else "eot"
    runs = {"oblivious": {"scores": [], "similarity": []}}
    for weight in weights:
        runs[f"{prefix}_w{weight:g}"] = {"scores": [], "similarity": []}

    for index, row in enumerate(pairs):
        source = Image.open(root / row["source_file"]).convert("RGB").resize(
            (160, 160), Image.BILINEAR
        )
        enroll_torch = get_embedding(
            Image.open(root / row["target_enroll_file"]).convert("RGB")
        )
        enroll_vector = enroll_torch.numpy().astype("float64")

        adversarial, similarity = generate_adversarial(
            source, enroll_torch, epsilon=0.03, n_steps=40, step_size=0.002
        )
        runs["oblivious"]["scores"].append(
            detector_score(adversarial, enroll_vector, embedder, features, statistics)
        )
        runs["oblivious"]["similarity"].append(similarity)

        for weight in weights:
            config = AdaptiveAttackConfig(consistency_weight=weight)
            adversarial, similarity = run_eot_attack(
                source,
                enroll_torch,
                config,
                transforms=attack_transforms,
                use_bpda=args.bpda,
            )
            key = f"{prefix}_w{weight:g}"
            runs[key]["scores"].append(
                detector_score(adversarial, enroll_vector, embedder, features, statistics)
            )
            runs[key]["similarity"].append(similarity)

        if (index + 1) % 5 == 0:
            print(f"  {index + 1}/{len(pairs)}", flush=True)

    print(f"\n표본 {len(pairs)}쌍\n")
    print(f"{'공격':<14} {'탐지율':>8} {'인증성공':>9} {'둘다':>7} {'점수중앙값':>11}")
    print("-" * 56)
    report = {"threshold": threshold, "n_pairs": len(pairs), "runs": {}}
    for name, values in runs.items():
        scores = np.asarray(values["scores"])
        similarity = np.asarray(values["similarity"])
        detected = scores >= threshold
        accepted = similarity >= args.identity_threshold
        bypass = accepted & ~detected
        report["runs"][name] = {
            "detection_rate": float(detected.mean()),
            "identity_accept_rate": float(accepted.mean()),
            "bypass_rate": float(bypass.mean()),
            "score_median": float(np.median(scores)),
        }
        print(
            f"{name:<14} {detected.mean():>8.1%} {accepted.mean():>9.1%} "
            f"{bypass.mean():>7.1%} {np.median(scores):>11.3f}"
        )
    print("\n둘다 = 인증 통과 + 탐지 회피 = 실제 공격 성공률")
    if args.bpda:
        print("\nBPDA: 미분 불가 변환의 backward를 항등으로 근사했다. forward는 방어가")
        print("실제로 계산하는 값과 같다. 근사가 완벽하지 않으므로 하한으로 본다.")
    else:
        print("\n주의: EOT는 미분 가능한 blur만 최적화한다. median과 JPEG 항은 손대지")
        print("못하므로 이 결과로 adaptive robustness를 주장할 수 없다.")

    if args.out:
        Path(args.out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
