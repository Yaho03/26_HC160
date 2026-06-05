"""Train a Denoising Autoencoder (DAE) for adversarial face purification.

학습 전략:
  - 입력: clean LFW 얼굴 이미지에 랜덤 Gaussian noise 추가 (σ ~ Uniform[0.01, 0.05])
  - 타겟: clean 이미지
  - 손실: MSE + perceptual similarity (선택)

adversarial perturbation과 학습 noise의 관계:
  FaceNet 공격 perturbation (eps=0.01 ~ Linf) ≈ 픽셀 ±2.55/255 ~ ±0.01
  랜덤 noise sigma=0.01~0.05 범위는 이 perturbation 크기를 포함함.
  따라서 DAE를 noise 제거로 학습하면 adversarial perturbation도 제거 가능.

추가 전략 (adv_dir 옵션):
  실제 adversarial 이미지가 있으면 (clean, adv) 쌍으로 직접 학습 — 더 강력한 정화 성능.

Colab 실행 예시:
  python -m src.training.train_face_dae \
    --data-dir data/raw/lfw \
    --out-dir checkpoints/face_dae \
    --epochs 30 \
    --batch-size 32
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm

from src.defenses.dae_model import build_dae

FACE_SIZE = (160, 160)


class LFWDenoisingDataset(Dataset):
    """LFW 이미지에 랜덤 noise를 추가해 (noisy, clean) 쌍으로 반환."""

    def __init__(
        self,
        data_dir: Path,
        noise_sigma_min: float = 0.01,
        noise_sigma_max: float = 0.05,
        adv_dir: Path | None = None,
    ) -> None:
        self.to_tensor = transforms.Compose([
            transforms.Resize(FACE_SIZE),
            transforms.ToTensor(),
        ])
        self.noise_sigma_min = noise_sigma_min
        self.noise_sigma_max = noise_sigma_max

        # Clean LFW 이미지 목록
        exts = {".jpg", ".jpeg", ".png"}
        self.clean_paths = [
            p for p in sorted(data_dir.rglob("*"))
            if p.suffix.lower() in exts and p.is_file()
        ]

        # (clean, adv) 쌍 — adv_dir에서 attack_handoff_index.csv 읽거나 이미지 직접 매핑
        self.adv_pairs: list[tuple[Path, Path]] = []
        if adv_dir and adv_dir.exists():
            index_csv = adv_dir / "attack_handoff_index.csv"
            if index_csv.exists():
                import csv
                with index_csv.open(newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        src = Path(row.get("source_file", ""))
                        adv = Path(row.get("adv_file", ""))
                        if src.exists() and adv.exists():
                            self.adv_pairs.append((src, adv))

        if not self.clean_paths:
            raise FileNotFoundError(f"LFW 이미지를 {data_dir}에서 찾을 수 없음")

    def __len__(self) -> int:
        return len(self.clean_paths) + len(self.adv_pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < len(self.clean_paths):
            # Noise augmentation: clean → noisy
            clean = self.to_tensor(Image.open(self.clean_paths[idx]).convert("RGB"))
            sigma = random.uniform(self.noise_sigma_min, self.noise_sigma_max)
            noisy = (clean + sigma * torch.randn_like(clean)).clamp(0, 1)
            return noisy, clean
        else:
            # Real adversarial pair
            pair_idx = idx - len(self.clean_paths)
            src_path, adv_path = self.adv_pairs[pair_idx]
            clean = self.to_tensor(Image.open(src_path).convert("RGB"))
            adv = self.to_tensor(Image.open(adv_path).convert("RGB"))
            return adv, clean


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for noisy, clean in tqdm(loader, desc="train", leave=False):
        noisy, clean = noisy.to(device), clean.to(device)
        pred = model(noisy)
        loss = criterion(pred, clean)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * noisy.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        pred = model(noisy)
        total_loss += criterion(pred, clean).item() * noisy.size(0)
    return total_loss / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="DAE 학습: adversarial face 정화 autoencoder")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/lfw"),
                        help="LFW 이미지 루트 디렉토리")
    parser.add_argument("--adv-dir", type=Path, default=None,
                        help="adversarial handoff 패키지 경로 (optional). 있으면 real adv pairs도 학습에 포함")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/face_dae"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-ch", type=int, default=32,
                        help="DAE base channel 수. 32=균형, 16=가벼움, 64=강력하지만 느림")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--noise-sigma-min", type=float, default=0.01)
    parser.add_argument("--noise-sigma-max", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = LFWDenoisingDataset(
        args.data_dir,
        noise_sigma_min=args.noise_sigma_min,
        noise_sigma_max=args.noise_sigma_max,
        adv_dir=args.adv_dir,
    )
    print(f"Dataset: {len(dataset)} samples "
          f"({len(dataset.clean_paths)} clean noise pairs, "
          f"{len(dataset.adv_pairs)} real adv pairs)")

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_dae(args.base_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = val_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.perf_counter() - t0

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "base_ch": args.base_ch,
            }, args.out_dir / "best.pt")

        marker = " ← best" if improved else ""
        print(f"[{epoch:3d}/{args.epochs}] train={train_loss:.6f}  val={val_loss:.6f}  "
              f"lr={scheduler.get_last_lr()[0]:.6f}  {elapsed:.1f}s{marker}")

    # 마지막 체크포인트
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
        "base_ch": args.base_ch,
    }, args.out_dir / "last.pt")

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Checkpoints: {args.out_dir.resolve()}")

    # 학습 history CSV 저장
    import csv
    hist_path = args.out_dir / "training_history.csv"
    with hist_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)
    print(f"History: {hist_path}")


if __name__ == "__main__":
    main()
