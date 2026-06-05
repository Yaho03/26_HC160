"""Verification defense: DiffPure-style diffusion purification.

DiffPure 핵심 아이디어:
  adversarial perturbation은 자연 이미지 분포에서 벗어난 고주파 노이즈.
  diffusion forward process로 적은 noise를 추가(t_diff 단계)한 뒤,
  reverse denoising으로 자연 이미지 분포로 복원 → perturbation이 제거됨.

구현 전략:
  1. HuggingFace diffusers의 pretrained DDPM 사용 (google/ddpm-celebahq-256 권장)
     - CelebA-HQ 256×256에서 학습된 모델로 얼굴 분포를 잘 알고 있음
     - 입력/출력: 256×256 → FaceNet용 160×160으로 resize
  2. diffusers 없을 때: noise-add + classical denoising (proxy fallback)

SDEdit / DiffPure 파라미터:
  t_diff: 전체 T step 중 noise를 추가할 비율 (0.05~0.15 권장)
    - 너무 작으면 perturbation 제거 불충분
    - 너무 크면 얼굴 identity 손상 → FRR 증가

Colab 실행 예시:
  # HuggingFace DDPM 사용:
  python -m src.defenses.verification_defense_diffpure \
    --handoff-index outputs/handoff/facenet_pgd_png_package/attack_handoff_index.csv \
    --model-id google/ddpm-celebahq-256 \
    --t-diff 0.10 \
    --out-dir outputs/defenses/verification/diffpure \
    --pretrained vggface2

  # Fallback (diffusers 없이 proxy 사용):
  python -m src.defenses.verification_defense_diffpure \
    --handoff-index ... \
    --use-fallback \
    --out-dir ...
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.verification.facenet_utils import (
    build_facenet_model,
    cosine_score,
    facenet_embedding,
    facenet_pixel_transform,
    load_facenet_image,
)

FACE_SIZE = (160, 160)
DDPM_SIZE = (256, 256)


# ------------------------------------------------------------------
# DiffPure core
# ------------------------------------------------------------------

def build_diffpure_model(model_id: str, device: torch.device):
    """diffusers DDPM 로드. google/ddpm-celebahq-256 권장."""
    try:
        from diffusers import DDPMPipeline
    except ImportError as e:
        raise ImportError(
            "diffusers 패키지가 필요합니다. Colab에서: pip install diffusers"
        ) from e
    pipe = DDPMPipeline.from_pretrained(model_id).to(device)
    pipe.unet.eval()
    return pipe


def diffpure(
    x: torch.Tensor,
    pipe,
    t_diff: float,
    device: torch.device,
) -> torch.Tensor:
    """DiffPure: forward diffuse to t, then reverse denoise back to 0.

    x: [1, 3, H, W] 픽셀 텐서 [0, 1]
    t_diff: noise 비율 (0.05~0.15)
    반환: purified 텐서 [1, 3, H, W] [0, 1]
    """
    # 256×256으로 resize
    x_256 = F.interpolate(x, size=DDPM_SIZE, mode="bilinear", align_corners=False)
    # [0,1] → [-1,1] (diffusers 표준)
    x_scaled = x_256 * 2.0 - 1.0

    scheduler = pipe.scheduler
    T = scheduler.config.num_train_timesteps
    t = int(t_diff * T)
    t_tensor = torch.tensor([t], device=device)

    # Forward: add noise at step t
    noise = torch.randn_like(x_scaled)
    noisy = scheduler.add_noise(x_scaled, noise, t_tensor)

    # Reverse: denoise from t back to 0
    scheduler.set_timesteps(T)
    denoised = noisy.clone()
    for timestep in scheduler.timesteps:
        if timestep > t:
            continue
        with torch.no_grad():
            noise_pred = pipe.unet(denoised, timestep).sample
        denoised = scheduler.step(noise_pred, timestep, denoised).prev_sample

    # [-1,1] → [0,1]
    purified_256 = (denoised.clamp(-1, 1) + 1.0) / 2.0
    # 160×160으로 resize back
    return F.interpolate(purified_256, size=FACE_SIZE, mode="bilinear", align_corners=False).clamp(0, 1)


# ------------------------------------------------------------------
# Fallback: noise-add + Gaussian denoising (DiffPure proxy)
# ------------------------------------------------------------------

def _gaussian_kernel_1d(sigma: float, size: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_blur_tensor(x: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    size = max(3, int(6 * sigma + 1)) | 1  # 홀수 보장
    k1d = _gaussian_kernel_1d(sigma, size, x.device)
    k2d = k1d.unsqueeze(1) * k1d.unsqueeze(0)
    k2d = k2d.unsqueeze(0).unsqueeze(0).expand(x.shape[1], 1, -1, -1).contiguous()
    return F.conv2d(x, k2d, padding=size // 2, groups=x.shape[1]).clamp(0, 1)


def diffpure_fallback(x: torch.Tensor, noise_sigma: float = 0.03, denoise_sigma: float = 1.5) -> torch.Tensor:
    """DiffPure proxy: noise 추가 + Gaussian denoising.

    실제 diffusion 모델 없이 개념만 시험하거나,
    Gaussian blur adaptive attack에서 Colab 비용을 절약할 때 사용.
    """
    noisy = (x + noise_sigma * torch.randn_like(x)).clamp(0, 1)
    return _gaussian_blur_tensor(noisy, sigma=denoise_sigma)


# ------------------------------------------------------------------
# Shared evaluation helpers
# ------------------------------------------------------------------

def load_handoff(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float_safe(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return default


def save_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    transforms.ToPILImage()(tensor.squeeze(0).cpu()).save(path)


def evaluate_defense(
    rows_in: list[dict[str, str]],
    purify_fn,
    defense_name: str,
    defense_params: dict,
    handoff_root: Path,
    facenet,
    to_pixel,
    device: torch.device,
    images_dir: Path,
    save_images: bool,
) -> list[dict[str, object]]:
    to_tensor = transforms.Compose([
        transforms.Resize(FACE_SIZE),
        transforms.ToTensor(),
    ])

    rows_out = []
    for row in tqdm(rows_in, desc=f"{defense_name} defense"):
        adv_path = Path(row["adv_file"])
        if not adv_path.is_absolute():
            adv_path = handoff_root / adv_path

        target_path = Path(row["target_enroll_file"])
        if not target_path.is_absolute():
            target_path = handoff_root / target_path

        if not adv_path.exists():
            rows_out.append({**row, "status": "missing_adv_file", "defense": defense_name})
            continue

        start = time.perf_counter()
        try:
            adv_tensor = to_tensor(Image.open(adv_path).convert("RGB")).unsqueeze(0).to(device)
            purified = purify_fn(adv_tensor)
        except Exception as e:
            rows_out.append({**row, "status": f"purify_error:{e}", "defense": defense_name})
            continue

        if save_images:
            defended_path = images_dir / f"{row['sample_id']}_{defense_name}.png"
            save_image(purified, defended_path)
        else:
            defended_path = Path("")

        with torch.no_grad():
            purified_emb = facenet_embedding(facenet, purified)
            target_tensor = load_facenet_image(target_path, to_pixel, device)
            target_emb = facenet_embedding(facenet, target_tensor)
            sim_after_defense = cosine_score(purified_emb, target_emb)

        elapsed = time.perf_counter() - start

        threshold = parse_float_safe(row.get("threshold", 0))
        sim_after_attack = parse_float_safe(row.get("similarity_after_attack", 0))
        accepted_after_attack = sim_after_attack >= threshold
        accepted_after_defense = sim_after_defense >= threshold
        attack_success_before_defense = parse_bool(row.get("attack_success_before_defense", False))
        defense_success = attack_success_before_defense and not accepted_after_defense

        rows_out.append({
            "sample_id": row["sample_id"],
            "pair_id": row.get("pair_id", ""),
            "attack": row.get("attack", ""),
            "defense": defense_name,
            "defense_params": json.dumps(defense_params),
            "model": row.get("model", ""),
            "pretrained": row.get("pretrained", ""),
            "source_file": row.get("source_file", ""),
            "target_enroll_file": row.get("target_enroll_file", ""),
            "adv_file": str(adv_path),
            "defended_file": str(defended_path),
            "source_name": row.get("source_name", ""),
            "target_name": row.get("target_name", ""),
            "threshold": threshold,
            "similarity_before": row.get("similarity_before", ""),
            "similarity_after_attack": sim_after_attack,
            "similarity_after_defense": sim_after_defense,
            "accepted_before": row.get("accepted_before", ""),
            "accepted_after_attack": accepted_after_attack,
            "accepted_after_defense": accepted_after_defense,
            "attack_success_before_defense": attack_success_before_defense,
            "defense_success": defense_success,
            "epsilon": row.get("epsilon", ""),
            "alpha": row.get("alpha", ""),
            "steps": row.get("steps", ""),
            "l2": row.get("l2", ""),
            "linf": row.get("linf", ""),
            "defense_time_sec": elapsed,
            "status": "ok",
        })
    return rows_out


def print_summary(rows_out: list[dict], defense_name: str, result_path: Path) -> None:
    ok_rows = [r for r in rows_out if r.get("status") == "ok"]
    attacked_rows = [r for r in ok_rows if parse_bool(r.get("attack_success_before_defense", False))]
    if attacked_rows:
        dsr = sum(parse_bool(r["defense_success"]) for r in attacked_rows) / len(attacked_rows)
        still = sum(parse_bool(r["accepted_after_defense"]) for r in attacked_rows) / len(attacked_rows)
        avg_drop = sum(
            parse_float_safe(r["similarity_after_attack"]) - parse_float_safe(r["similarity_after_defense"])
            for r in attacked_rows
        ) / len(attacked_rows)
        avg_t = sum(parse_float_safe(r["defense_time_sec"]) for r in ok_rows) / max(len(ok_rows), 1)
        print(f"\n[{defense_name}] Rows={len(rows_out)} ok={len(ok_rows)} attacked={len(attacked_rows)}")
        print(f"  Defense success rate: {dsr:.2%}")
        print(f"  Still accepted (ASR): {still:.2%}")
        print(f"  Avg similarity drop:  {avg_drop:.4f}")
        print(f"  Avg defense time:     {avg_t:.4f}s")
    print(f"  Saved: {result_path}")


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DiffPure verification defense.")
    parser.add_argument("--handoff-index", type=Path,
                        default=Path("outputs/handoff/facenet_pgd_png_package/attack_handoff_index.csv"))
    parser.add_argument("--handoff-root", type=Path, default=None)
    parser.add_argument("--pretrained", default="vggface2", choices=["vggface2", "casia-webface"])
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/defenses/verification/diffpure"))
    parser.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True)

    # DiffPure 설정
    parser.add_argument("--use-fallback", action="store_true",
                        help="diffusers 없이 noise+blur proxy를 사용 (빠르지만 정확도 낮음).")
    parser.add_argument("--model-id", default="google/ddpm-celebahq-256",
                        help="HuggingFace DDPM 모델 ID.")
    parser.add_argument("--t-diff", type=float, default=0.10,
                        help="Forward diffusion 비율 (0.05~0.15). 크면 방어↑ identity 손상↑.")

    # Fallback 설정
    parser.add_argument("--fallback-noise-sigma", type=float, default=0.03)
    parser.add_argument("--fallback-denoise-sigma", type=float, default=1.5)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    facenet, _ = build_facenet_model(args.pretrained)
    to_pixel = facenet_pixel_transform()
    rows_in = load_handoff(args.handoff_index)
    handoff_root = args.handoff_root or args.handoff_index.parent
    images_dir = args.out_dir / "images"

    if args.use_fallback:
        defense_name = "diffpure_fallback"
        defense_params = {
            "noise_sigma": args.fallback_noise_sigma,
            "denoise_sigma": args.fallback_denoise_sigma,
        }
        def purify_fn(x: torch.Tensor) -> torch.Tensor:
            return diffpure_fallback(x, args.fallback_noise_sigma, args.fallback_denoise_sigma)
    else:
        defense_name = f"diffpure_t{args.t_diff:.2f}"
        defense_params = {"model_id": args.model_id, "t_diff": args.t_diff}
        pipe = build_diffpure_model(args.model_id, device)
        def purify_fn(x: torch.Tensor) -> torch.Tensor:
            return diffpure(x, pipe, args.t_diff, device)

    rows_out = evaluate_defense(
        rows_in, purify_fn, defense_name, defense_params,
        handoff_root, facenet, to_pixel, device,
        images_dir, args.save_images,
    )

    result_path = args.out_dir / f"verification_defense_{defense_name}.csv"
    save_csv(rows_out, result_path)
    print_summary(rows_out, defense_name, result_path)


if __name__ == "__main__":
    main()
