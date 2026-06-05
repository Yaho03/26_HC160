"""Verification defense: Denoising Autoencoder (DAE) purification.

학습된 FaceDAE 체크포인트를 사용해 adversarial face image에서
perturbation을 제거하고, FaceNet similarity를 재계산하여 방어 성공 여부를 평가한다.

방어팀 handoff 포맷(attack_handoff_index.csv)을 읽어서 같은 포맷으로 결과를 출력한다.

Colab 실행 예시:
  python -m src.defenses.verification_defense_dae \
    --handoff-index outputs/handoff/facenet_pgd_png_package/attack_handoff_index.csv \
    --checkpoint checkpoints/face_dae/best.pt \
    --out-dir outputs/defenses/verification/dae \
    --pretrained vggface2
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.defenses.dae_model import load_dae
from src.verification.facenet_utils import (
    build_facenet_model,
    cosine_score,
    facenet_embedding,
    facenet_pixel_transform,
    load_facenet_image,
)

FACE_SIZE = (160, 160)


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


def purify_image(
    adv_path: Path,
    dae: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """adv 이미지 → DAE 정화 → 160×160 픽셀 텐서 [0,1] 반환."""
    to_tensor = transforms.Compose([
        transforms.Resize(FACE_SIZE),
        transforms.ToTensor(),
    ])
    adv_tensor = to_tensor(Image.open(adv_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        purified = dae(adv_tensor)
    return purified.clamp(0, 1)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    transforms.ToPILImage()(tensor.squeeze(0).cpu()).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="DAE purification defense for FaceNet verification attacks.")
    parser.add_argument("--handoff-index", type=Path,
                        default=Path("outputs/handoff/facenet_pgd_png_package/attack_handoff_index.csv"))
    parser.add_argument("--handoff-root", type=Path, default=None,
                        help="handoff 패키지 루트. 지정하면 adv_file 경로를 이 경로 기준으로 해석.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/face_dae/best.pt"))
    parser.add_argument("--pretrained", default="vggface2", choices=["vggface2", "casia-webface"])
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/defenses/verification/dae"))
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dae = load_dae(str(args.checkpoint), device, args.base_ch)
    print(f"DAE loaded from: {args.checkpoint}")

    facenet, _ = build_facenet_model(args.pretrained)
    to_pixel = facenet_pixel_transform()

    rows_in = load_handoff(args.handoff_index)
    handoff_root = args.handoff_root or args.handoff_index.parent

    images_dir = args.out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict[str, object]] = []

    for row in tqdm(rows_in, desc="DAE defense"):
        adv_path = Path(row["adv_file"])
        if not adv_path.is_absolute():
            adv_path = handoff_root / adv_path

        target_path = Path(row["target_enroll_file"])
        if not target_path.is_absolute():
            target_path = handoff_root / target_path

        if not adv_path.exists():
            rows_out.append({**row, "status": "missing_adv_file", "defense": "dae"})
            continue

        start = time.perf_counter()
        try:
            purified = purify_image(adv_path, dae, device)
        except Exception as e:
            rows_out.append({**row, "status": f"dae_error:{e}", "defense": "dae"})
            continue

        if args.save_images:
            defended_path = images_dir / f"{row['sample_id']}_dae.png"
            save_image(purified, defended_path)
        else:
            defended_path = Path("")

        # FaceNet similarity 재계산
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
            "defense": "dae",
            "defense_params": json.dumps({"checkpoint": str(args.checkpoint), "base_ch": args.base_ch}),
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

    result_path = args.out_dir / "verification_defense_dae.csv"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    ok_rows = [r for r in rows_out if r.get("status") == "ok"]
    attacked_rows = [r for r in ok_rows if parse_bool(r.get("attack_success_before_defense", False))]
    if attacked_rows:
        defense_success_rate = sum(parse_bool(r["defense_success"]) for r in attacked_rows) / len(attacked_rows)
        still_accepted = sum(parse_bool(r["accepted_after_defense"]) for r in attacked_rows) / len(attacked_rows)
        avg_sim_drop = sum(
            parse_float_safe(r["similarity_after_attack"]) - parse_float_safe(r["similarity_after_defense"])
            for r in attacked_rows
        ) / len(attacked_rows)
        avg_time = sum(parse_float_safe(r["defense_time_sec"]) for r in ok_rows) / max(len(ok_rows), 1)

        print(f"\nRows: {len(rows_out)}  ok={len(ok_rows)}  attacked={len(attacked_rows)}")
        print(f"Defense success rate:   {defense_success_rate:.2%}")
        print(f"Still accepted (ASR):   {still_accepted:.2%}")
        print(f"Avg similarity drop:    {avg_sim_drop:.4f}")
        print(f"Avg defense time:       {avg_time:.4f}s")
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
