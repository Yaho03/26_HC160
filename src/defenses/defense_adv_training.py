"""Adversarial Training defense.

clean 이미지 + PGD 적대적 이미지를 섞어 ResNet-50 을 fine-tune 한 뒤,
기존 attack_index.csv 의 adv_file 을 새 모델로 재분류하여 결과 CSV 를 저장한다.
결과 포맷은 defense_jpeg.py 등 다른 방어 스크립트와 동일하다.

Usage:
    python -m src.defenses.defense_adv_training \
        --checkpoint   checkpoints/face_resnet50_lfw10/best.pt \
        --data-dir     data/processed/lfw_identity_10 \
        --attack-index outputs/attacks/attack_index.csv \
        --out-dir      outputs/defenses/adv_training
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

from src.common.device import get_device

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


class AdvDataset(Dataset):
    """attack_index.csv 의 adv_file 을 true_label 로 학습하는 Dataset."""

    def __init__(self, adv_df: pd.DataFrame, transform) -> None:
        mask = adv_df["adv_file"].apply(lambda p: Path(p).exists())
        self.df = adv_df[mask].reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = self.transform(Image.open(row["adv_file"]).convert("RGB"))
        return image, int(row["true_label"])


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss    += float(loss.detach().cpu()) * labels.size(0)
            total_correct += int((logits.argmax(1) == labels).sum().detach().cpu())
            total         += labels.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    type=Path, default=Path("checkpoints/face_resnet50_lfw10/best.pt"))
    parser.add_argument("--data-dir",      type=Path, default=Path("data/processed/lfw_identity_10"))
    parser.add_argument("--attack-index",  type=Path, default=Path("outputs/attacks/attack_index.csv"))
    parser.add_argument("--attack-family", type=str,  default="pgd")
    parser.add_argument("--out-dir",       type=Path, default=Path("outputs/defenses/adv_training"))
    parser.add_argument("--epochs",        type=int,  default=5)
    parser.add_argument("--batch-size",    type=int,  default=32)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--mix-ratio",     type=float, default=0.5)
    parser.add_argument("--num-workers",   type=int,  default=2)
    parser.add_argument("--limit",         type=int,  default=0)
    parser.add_argument("--drive-cache",   type=Path, default=None,
                        help="Drive 캐시 경로. best_adv.pt 가 있으면 학습 생략.")
    args = parser.parse_args()

    device = get_device()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    best_path = args.out_dir / "best_adv.pt"

    # ── Drive 캐시 확인 → 학습 생략 ──────────────────
    if args.drive_cache and (args.drive_cache / "best_adv.pt").exists():
        print(f"[INFO] Drive 캐시 발견 → 학습 생략: {args.drive_cache / 'best_adv.pt'}")
        import shutil
        shutil.copy(args.drive_cache / "best_adv.pt", best_path)
        skip_training = True
    elif best_path.exists():
        print(f"[INFO] 로컬 체크포인트 발견 → 학습 생략: {best_path}")
        skip_training = True
    else:
        skip_training = False

    # ── 모델 로드 ─────────────────────────────────
    src_ckpt = best_path if skip_training else args.checkpoint
    ckpt = torch.load(src_ckpt, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    model = build_model(len(classes)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # ── Transform ─────────────────────────────────
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # ── Mixed Dataset (clean + adv) ───────────────
    attack_index = pd.read_csv(args.attack_index)
    adv_df = attack_index[
        (attack_index["attack_family"] == args.attack_family) &
        bool_series(attack_index["clean_correct"]) &
        bool_series(attack_index["success_on_clean"])
    ][["adv_file", "true_label"]].copy()

    clean_ds = datasets.ImageFolder(args.data_dir / "train", transform=train_tf)
    target_n = int(len(clean_ds) * args.mix_ratio / (1 - args.mix_ratio))
    adv_df = adv_df.sample(n=min(target_n, len(adv_df)), replace=target_n > len(adv_df),
                           random_state=42).reset_index(drop=True)
    adv_train_ds = AdvDataset(adv_df, train_tf)

    train_loader = DataLoader(ConcatDataset([clean_ds, adv_train_ds]),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader   = DataLoader(datasets.ImageFolder(args.data_dir / "val", eval_tf),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    # ── Fine-tuning ───────────────────────────────
    if not skip_training:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        best_val_acc = -1.0

        epoch_bar = tqdm(range(1, args.epochs + 1), desc="Fine-tuning", unit="epoch")
        for epoch in epoch_bar:
            train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, device)
            epoch_bar.set_postfix(
                train_acc=f"{train_acc:.2%}",
                val_acc=f"{val_acc:.2%}",
                best=f"{best_val_acc:.2%}",
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({"model_state_dict": model.state_dict(),
                            "classes": classes}, best_path)
                epoch_bar.write(f"  epoch {epoch:02d}  val_acc={val_acc:.2%}  → best 저장")

        # Drive 캐시 저장
        if args.drive_cache:
            args.drive_cache.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(best_path, args.drive_cache / "best_adv.pt")
            print(f"[INFO] Drive 캐시 저장: {args.drive_cache / 'best_adv.pt'}")

    # ── 평가: adv 이미지를 새 모델로 재분류 ──────────
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mean_t = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor(STD,  device=device).view(1, 3, 1, 1)
    to_tensor = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    eval_df = attack_index[
        bool_series(attack_index["clean_correct"]) &
        bool_series(attack_index["success_on_clean"])
    ].copy()
    if args.limit > 0:
        eval_df = eval_df.head(args.limit)

    rows = []
    for _, attack in tqdm(eval_df.iterrows(), total=len(eval_df), desc="adv_training eval"):
        adv_path = Path(str(attack["adv_file"]))
        if not adv_path.exists():
            rows.append({
                "sample_id": attack["sample_id"],
                "attack_family": attack["attack_family"],
                "attack": attack["attack"],
                "defense": "adv_training",
                "defense_params": json.dumps({"attack_family": args.attack_family,
                                              "mix_ratio": args.mix_ratio,
                                              "epochs": args.epochs}),
                "input_adv_file": str(adv_path),
                "defended_file": "",
                "pred_before_defense": attack.get("pred_after", ""),
                "pred_after_defense": "",
                "pred_after_defense_name": "",
                "target_label": attack["target_label"],
                "true_label": attack["true_label"],
                "attack_success_before_defense": attack["success_on_clean"],
                "attack_success_after_defense": "",
                "recovered": "",
                "target_conf_before_defense": attack.get("target_conf_after", ""),
                "target_conf_after_defense": "",
                "true_conf_after_defense": "",
                "defense_time_sec": "",
                "status": "missing_adv_file",
            })
            continue

        start = time.perf_counter()
        tensor = to_tensor(Image.open(adv_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(model((tensor - mean_t) / std_t), dim=1)
            pred = int(probs.argmax(1).item())
            target_label = int(attack["target_label"])
            true_label   = int(attack["true_label"])
        elapsed = time.perf_counter() - start

        rows.append({
            "sample_id": attack["sample_id"],
            "attack_family": attack["attack_family"],
            "attack": attack["attack"],
            "defense": "adv_training",
            "defense_params": json.dumps({"attack_family": args.attack_family,
                                          "mix_ratio": args.mix_ratio,
                                          "epochs": args.epochs}),
            "input_adv_file": str(adv_path),
            "defended_file": str(best_path),
            "pred_before_defense": attack.get("pred_after", ""),
            "pred_after_defense": pred,
            "pred_after_defense_name": classes[pred],
            "target_label": target_label,
            "true_label": true_label,
            "attack_success_before_defense": attack["success_on_clean"],
            "attack_success_after_defense": pred == target_label,
            "recovered": pred == true_label,
            "target_conf_before_defense": attack.get("target_conf_after", ""),
            "target_conf_after_defense": float(probs[0, target_label].cpu()),
            "true_conf_after_defense": float(probs[0, true_label].cpu()),
            "defense_time_sec": elapsed,
            "status": "ok",
        })

    result_path = args.out_dir / "adv_training_results.csv"
    with result_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    before = [r for r in ok if str(r["attack_success_before_defense"]).lower() in {"true","1"}]
    defense_rate = 1 - sum(bool(r["attack_success_after_defense"]) for r in before) / len(before) if before else 0
    recovery_rate = sum(bool(r["recovered"]) for r in before) / len(before) if before else 0
    print(f"Defense Success Rate: {defense_rate:.2%}")
    print(f"Recovery Rate:        {recovery_rate:.2%}")
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
