"""
FaceNet Adversarial Training Defense (3단계)

(source, adv, target_enroll) 트리플릿으로 InceptionResnetV1을 fine-tune한다.
adv embedding이 source에 가까워지고 target에서 멀어지도록 학습.

학습 전략:
  - block8 + last_linear + last_bn만 unfreeze (나머지 freeze)
  - Loss: Triplet margin loss (cosine similarity 기반)
      max(0, cos(adv, target) - cos(adv, source) + margin)
  - Epoch 5, LR 1e-5, batch 8

실행:
    python -m src.verification.defenses.verification_defense_adv_training \\
        --index  /path/to/attack_handoff_index.csv \\
        --pkg-root /path/to/facenet_verification_attack_package \\
        --out-dir outputs/verification_defense/adv_training
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from facenet_pytorch import InceptionResnetV1

THRESHOLD = 0.47966246581077576
DEFAULT_MARGIN  = 0.15
DEFAULT_EPOCHS  = 5
DEFAULT_LR      = 1e-5
DEFAULT_BATCH   = 8


# ── 전처리 ────────────────────────────────────────────────────────────────────

def preprocess(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((160, 160), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = (arr - 127.5) / 128.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3, 160, 160)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TripletDataset(Dataset):
    """(adv, source, target_enroll) 트리플릿 Dataset."""

    def __init__(self, rows: list[dict], pkg_root: str, only_success: bool = True):
        if only_success:
            rows = [r for r in rows
                    if str(r.get("accepted_after_attack", "False")).lower() == "true"]
        self.rows     = rows
        self.pkg_root = pkg_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        adv    = preprocess(Image.open(os.path.join(self.pkg_root, r["adv_file"])))
        source = preprocess(Image.open(os.path.join(self.pkg_root, r["source_file"])))
        target = preprocess(Image.open(os.path.join(self.pkg_root, r["target_enroll_file"])))
        return adv, source, target, r["sample_id"]


# ── 모델 준비 ─────────────────────────────────────────────────────────────────

def load_model(device: torch.device) -> InceptionResnetV1:
    model = InceptionResnetV1(pretrained="vggface2").to(device)
    # block8, last_linear, last_bn만 학습 — 나머지 freeze
    for name, param in model.named_parameters():
        layer = name.split(".")[0]
        param.requires_grad = layer in ("block8", "last_linear", "last_bn")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"학습 파라미터: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    return model


def get_embedding(model: InceptionResnetV1, tensor: torch.Tensor) -> torch.Tensor:
    """(B, 3, 160, 160) → L2 정규화된 (B, 512) embedding"""
    emb = model(tensor)
    return F.normalize(emb, p=2, dim=1)


# ── Loss ──────────────────────────────────────────────────────────────────────

def triplet_cosine_loss(
    adv_emb: torch.Tensor,
    source_emb: torch.Tensor,
    target_emb: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """
    cos(adv, target) - cos(adv, source) + margin 을 최소화.
    adv가 source에 가까워지고 target에서 멀어지도록 학습.
    """
    pos_sim = F.cosine_similarity(adv_emb, source_emb, dim=1)  # (B,) 높을수록 good
    neg_sim = F.cosine_similarity(adv_emb, target_emb, dim=1)  # (B,) 낮을수록 good
    loss = F.relu(neg_sim - pos_sim + margin)
    return loss.mean()


# ── 평가 ──────────────────────────────────────────────────────────────────────

def evaluate(model: InceptionResnetV1, loader: DataLoader, device: torch.device, threshold: float):
    model.eval()
    n_total = n_accepted = 0
    with torch.no_grad():
        for adv, source, target, _ in loader:
            adv_emb    = get_embedding(model, adv.to(device))
            target_emb = get_embedding(model, target.to(device))
            sims = F.cosine_similarity(adv_emb, target_emb, dim=1)
            n_accepted += int((sims >= threshold).sum())
            n_total    += adv.size(0)
    asr = n_accepted / n_total if n_total else 0.0
    return asr, n_accepted, n_total


# ── 학습 루프 ─────────────────────────────────────────────────────────────────

def train(
    index_csv: str,
    pkg_root: str,
    out_dir: str,
    epochs: int   = DEFAULT_EPOCHS,
    lr: float     = DEFAULT_LR,
    batch: int    = DEFAULT_BATCH,
    margin: float = DEFAULT_MARGIN,
    threshold: float = THRESHOLD,
    device_str: str  = "auto",
) -> str:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if device_str == "auto" else device_str
    )
    print(f"device: {device}")

    rows = list(csv.DictReader(open(index_csv)))
    dataset = TripletDataset(rows, pkg_root, only_success=True)
    loader  = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=0)

    model     = load_model(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4,
    )

    # ── 학습 전 ASR 측정 ──────────────────────────────────────────────────────
    asr_before, n_acc_before, n_total = evaluate(model, loader, device, threshold)
    print(f"\n[학습 전] ASR: {asr_before:.1%}  ({n_acc_before}/{n_total})")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    history = []
    best_asr = asr_before
    best_ckpt = str(out_path / "best_adv_trained.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for adv, source, target, _ in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            adv_emb    = get_embedding(model, adv.to(device))
            source_emb = get_embedding(model, source.to(device)).detach()
            target_emb = get_embedding(model, target.to(device)).detach()

            loss = triplet_cosine_loss(adv_emb, source_emb, target_emb, margin)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        asr, n_acc, _ = evaluate(model, loader, device, threshold)

        print(f"  Epoch {epoch:02d}  loss={avg_loss:.4f}  ASR={asr:.1%} ({n_acc}/{n_total})")
        history.append({"epoch": epoch, "loss": avg_loss, "asr": asr})

        if asr < best_asr:
            best_asr = asr
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "asr": asr,
                "threshold": threshold,
            }, best_ckpt)
            print(f"    → best checkpoint 저장 (ASR {asr:.1%})")

    # ── 최종 평가 ─────────────────────────────────────────────────────────────
    print(f"\n=== Adversarial Training 완료 ===")
    print(f"  학습 전 ASR: {asr_before:.1%} ({n_acc_before}/{n_total})")
    print(f"  학습 후 ASR: {best_asr:.1%}  (best checkpoint 기준)")

    # ── best 모델로 샘플별 결과 CSV 저장 ─────────────────────────────────────
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_rows = []
    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    with torch.no_grad():
        for (adv, source, target, sample_ids) in tqdm(eval_loader, desc="최종 평가", leave=False):
            adv_emb    = get_embedding(model, adv.to(device))
            target_emb = get_embedding(model, target.to(device))
            source_emb = get_embedding(model, source.to(device))

            sim_adv_target = float(F.cosine_similarity(adv_emb, target_emb).item())
            sim_adv_source = float(F.cosine_similarity(adv_emb, source_emb).item())
            accepted_after = sim_adv_target >= threshold

            sid = sample_ids[0] if isinstance(sample_ids, (list, tuple)) else sample_ids
            orig = next(r for r in rows if r["sample_id"] == sid)

            out_rows.append({
                "sample_id":               sid,
                "defense":                 "adv_training",
                "defense_params":          json.dumps({"epochs": ckpt["epoch"], "lr": lr, "margin": margin}),
                "threshold":               threshold,
                "similarity_after_attack": orig.get("similarity_after_attack", ""),
                "sim_adv_target":          round(sim_adv_target, 8),
                "sim_adv_source":          round(sim_adv_source, 8),
                "accepted_after_attack":   orig.get("accepted_after_attack", ""),
                "accepted_after_defense":  accepted_after,
                "attack_success_after_defense": accepted_after,
                "defense_success":         str(orig.get("accepted_after_attack", "")).lower() == "true" and not accepted_after,
            })

    result_csv = out_path / "verification_defense_adv_training.csv"
    with open(result_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    n_def = sum(1 for r in out_rows if r["defense_success"] is True or str(r["defense_success"]).lower() == "true")
    n_atk = sum(1 for r in out_rows if str(r["accepted_after_attack"]).lower() == "true")

    print(f"  방어 성공: {n_def}/{n_atk} ({n_def/n_atk*100:.1f}%)" if n_atk else "  방어 성공: 0")
    print(f"  checkpoint: {best_ckpt}")
    print(f"  결과 CSV: {result_csv}")

    # 학습 이력 저장
    with open(out_path / "training_history.json", "w") as f:
        json.dump({
            "asr_before": asr_before,
            "asr_best":   best_asr,
            "epochs":     epochs,
            "lr":         lr,
            "margin":     margin,
            "history":    history,
        }, f, indent=2)

    return str(result_csv)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaceNet Adversarial Training Defense")
    parser.add_argument("--index",    required=True, help="attack_handoff_index.csv 경로")
    parser.add_argument("--pkg-root", required=True, help="패키지 루트 (samples/ 상위)")
    parser.add_argument("--out-dir",  default="outputs/verification_defense/adv_training")
    parser.add_argument("--epochs",   type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--lr",       type=float, default=DEFAULT_LR)
    parser.add_argument("--batch",    type=int,   default=DEFAULT_BATCH)
    parser.add_argument("--margin",   type=float, default=DEFAULT_MARGIN)
    args = parser.parse_args()

    train(
        index_csv=args.index,
        pkg_root=args.pkg_root,
        out_dir=args.out_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch=args.batch,
        margin=args.margin,
    )
