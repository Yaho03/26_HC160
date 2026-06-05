"""방어 기법이 정상 사용자 인증(FRR)에 미치는 영향 평가.

방어팀 보고서의 한계: "FRR 미측정" — 방어 기법이 적대적 이미지를 막는 것은 확인됐지만,
정상 사용자의 얼굴 인증도 같이 방해하는지는 평가하지 않음.

이 스크립트:
  - LFW positive pair(같은 사람 2장)에 각 방어 기법 적용
  - 방어 전/후 FaceNet similarity 비교
  - FRR 변화 계산: 방어가 정상 인증에 얼마나 영향을 주는가
  - 이상적인 방어: FRR 거의 변화 없음 + ASR은 크게 낮춤

사용법:
  python -m src.verification.evaluate_defense_frr \
    --pairs outputs/verification/lfw_test_pairs.csv \
    --metrics outputs/verification_facenet/verification_metrics.json \
    --defenses smoothing jpeg bitdepth dae diffpure_fallback \
    --dae-checkpoint checkpoints/face_dae/best.pt \
    --out-dir outputs/defenses/verification/frr_evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms
from tqdm import tqdm

from src.verification.evaluate_face_verification import parse_bool
from src.verification.facenet_utils import (
    build_facenet_model,
    cosine_score,
    facenet_embedding,
    facenet_pixel_transform,
    load_facenet_image,
)

FACE_SIZE = (160, 160)


# ------------------------------------------------------------------
# Defense transform 함수들 (PIL 기반 — 방어팀 구현과 동일)
# ------------------------------------------------------------------

def apply_jpeg(img: Image.Image, quality: int = 75) -> torch.Tensor:
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return _to_tensor(Image.open(buf).convert("RGB"))


def apply_smoothing(img: Image.Image, radius: float = 3.0) -> torch.Tensor:
    return _to_tensor(img.filter(ImageFilter.GaussianBlur(radius=radius)).convert("RGB"))


def apply_bitdepth(img: Image.Image, bits: int = 4) -> torch.Tensor:
    import numpy as np
    arr = np.array(img)
    shift = 8 - bits
    quantized = (arr >> shift) << shift
    return _to_tensor(Image.fromarray(quantized.astype("uint8")))


_to_pil = transforms.ToPILImage()
_to_tensor_fn = transforms.Compose([transforms.Resize(FACE_SIZE), transforms.ToTensor()])


def _to_tensor(img: Image.Image) -> torch.Tensor:
    return _to_tensor_fn(img)


def apply_dae(tensor: torch.Tensor, dae, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        return dae(tensor.unsqueeze(0).to(device)).squeeze(0).clamp(0, 1).cpu()


def apply_diffpure_fallback(tensor: torch.Tensor, noise_sigma: float = 0.03, denoise_sigma: float = 1.5) -> torch.Tensor:
    from src.defenses.verification_defense_diffpure import diffpure_fallback
    return diffpure_fallback(tensor.unsqueeze(0), noise_sigma, denoise_sigma).squeeze(0).cpu()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="방어 기법의 정상 사용자 FRR 영향 평가.")
    parser.add_argument("--pairs", type=Path, default=Path("outputs/verification/lfw_test_pairs.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("outputs/verification_facenet/verification_metrics.json"))
    parser.add_argument("--pretrained", default="vggface2", choices=["vggface2", "casia-webface"])
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/defenses/verification/frr_evaluation"))
    parser.add_argument("--limit", type=int, default=200, help="positive pair 최대 개수. 0=전체.")
    parser.add_argument(
        "--defenses", nargs="+",
        default=["smoothing", "jpeg", "bitdepth"],
        choices=["smoothing", "jpeg", "bitdepth", "dae", "diffpure_fallback"],
        help="평가할 방어 기법.",
    )
    # smoothing
    parser.add_argument("--smoothing-radius", type=float, default=3.0)
    # jpeg
    parser.add_argument("--jpeg-quality", type=int, default=75)
    # bitdepth
    parser.add_argument("--bitdepth-bits", type=int, default=4)
    # dae
    parser.add_argument("--dae-checkpoint", type=str, default=None)
    parser.add_argument("--dae-base-ch", type=int, default=32)
    # diffpure fallback
    parser.add_argument("--fallback-noise-sigma", type=float, default=0.03)
    parser.add_argument("--fallback-denoise-sigma", type=float, default=1.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    threshold = float(metrics["eer_threshold"])
    clean_frr = float(metrics.get("frr_at_eer_threshold", metrics.get("frr", 0)))
    print(f"Threshold: {threshold:.4f}  |  Clean FRR: {clean_frr:.2%}")

    model, _ = build_facenet_model(args.pretrained)
    to_pixel = facenet_pixel_transform()

    # DAE 로드 (필요 시)
    dae = None
    if "dae" in args.defenses:
        if not args.dae_checkpoint:
            print("[경고] --dae-checkpoint 없이 dae 방어 평가 불가. dae 건너뜀.")
            args.defenses = [d for d in args.defenses if d != "dae"]
        else:
            from src.defenses.dae_model import load_dae
            dae = load_dae(args.dae_checkpoint, device, args.dae_base_ch)

    with args.pairs.open(newline="", encoding="utf-8") as f:
        all_pairs = list(csv.DictReader(f))

    # positive pair만 (same_identity=True)
    positive_pairs = [p for p in all_pairs if parse_bool(p["same_identity"])]
    if args.limit > 0:
        positive_pairs = positive_pairs[:args.limit]
    if not positive_pairs:
        raise ValueError("positive pair가 없습니다.")
    print(f"Positive pairs: {len(positive_pairs)}")

    # ------------------------------------------------------------------
    # clean similarity 먼저 계산 (캐시)
    # ------------------------------------------------------------------
    emb_cache: dict[str, torch.Tensor] = {}

    def get_emb(path_str: str) -> torch.Tensor:
        if path_str not in emb_cache:
            t = load_facenet_image(Path(path_str), to_pixel, device)
            with torch.no_grad():
                emb_cache[path_str] = facenet_embedding(model, t)
        return emb_cache[path_str]

    clean_sims = []
    clean_accepts = []
    for pair in positive_pairs:
        sim = cosine_score(get_emb(pair["left_file"]), get_emb(pair["right_file"]))
        clean_sims.append(sim)
        clean_accepts.append(sim >= threshold)

    clean_frr_measured = 1.0 - (sum(clean_accepts) / len(clean_accepts))
    print(f"Measured clean FRR on these pairs: {clean_frr_measured:.2%}")

    # ------------------------------------------------------------------
    # 방어별 FRR 측정
    # ------------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    for defense_name in args.defenses:
        print(f"\n--- {defense_name} ---")
        defended_sims = []
        defended_accepts = []

        for pair in tqdm(positive_pairs, desc=defense_name, leave=False):
            left_path = Path(pair["left_file"])
            right_path = Path(pair["right_file"])

            # left 이미지에 방어 적용 → embedding
            left_img = Image.open(left_path).convert("RGB")

            if defense_name == "smoothing":
                left_t = apply_smoothing(left_img, args.smoothing_radius).to(device).unsqueeze(0)
            elif defense_name == "jpeg":
                left_t = apply_jpeg(left_img, args.jpeg_quality).to(device).unsqueeze(0)
            elif defense_name == "bitdepth":
                left_t = apply_bitdepth(left_img, args.bitdepth_bits).to(device).unsqueeze(0)
            elif defense_name == "dae" and dae is not None:
                left_t_cpu = apply_dae(
                    _to_tensor_fn(left_img), dae, device
                )
                left_t = left_t_cpu.unsqueeze(0).to(device)
            elif defense_name == "diffpure_fallback":
                left_t_cpu = apply_diffpure_fallback(
                    _to_tensor_fn(left_img),
                    args.fallback_noise_sigma,
                    args.fallback_denoise_sigma,
                )
                left_t = left_t_cpu.unsqueeze(0).to(device)
            else:
                continue

            with torch.no_grad():
                left_emb = facenet_embedding(model, left_t)
                right_emb = get_emb(pair["right_file"])
                sim = cosine_score(left_emb, right_emb)

            defended_sims.append(sim)
            defended_accepts.append(sim >= threshold)

        if not defended_sims:
            continue

        defended_frr = 1.0 - (sum(defended_accepts) / len(defended_accepts))
        frr_increase = defended_frr - clean_frr_measured
        avg_sim_drop = (sum(clean_sims) / len(clean_sims)) - (sum(defended_sims) / len(defended_sims))

        print(f"  FRR after {defense_name}: {defended_frr:.2%}  "
              f"(+{frr_increase:.2%} vs clean)  "
              f"avg_sim_drop={avg_sim_drop:.4f}")

        summary_rows.append({
            "defense": defense_name,
            "n_pairs": len(defended_sims),
            "clean_frr": clean_frr_measured,
            "defended_frr": defended_frr,
            "frr_increase": frr_increase,
            "avg_sim_clean": sum(clean_sims) / len(clean_sims),
            "avg_sim_defended": sum(defended_sims) / len(defended_sims),
            "avg_sim_drop": avg_sim_drop,
        })

        # 세부 CSV
        detail_rows = [
            {
                "pair_id": positive_pairs[i]["pair_id"],
                "sim_clean": clean_sims[i],
                "sim_defended": defended_sims[i],
                "accept_clean": clean_accepts[i],
                "accept_defended": defended_accepts[i],
                "frr_caused": clean_accepts[i] and not defended_accepts[i],
            }
            for i in range(len(defended_sims))
        ]
        detail_path = args.out_dir / f"frr_{defense_name}.csv"
        with detail_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)

    # 요약 CSV
    if summary_rows:
        summary_path = args.out_dir / "frr_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"\n=== FRR 영향 요약 ===")
        print(f"{'Defense':<22} {'Clean FRR':>10} {'After FRR':>10} {'FRR 증가':>10} {'sim_drop':>10}")
        print("-" * 65)
        for r in summary_rows:
            print(f"{r['defense']:<22} {r['clean_frr']:>10.2%} {r['defended_frr']:>10.2%} "
                  f"{r['frr_increase']:>+10.2%} {r['avg_sim_drop']:>10.4f}")
        print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
