"""Denoising Autoencoder (DAE) for adversarial face purification.

U-Net 구조: skip connection으로 얼굴 디테일(눈, 코, 입 위치)을 보존하면서
adversarial perturbation(고주파 노이즈)을 제거한다.

입력/출력: 3×160×160 (FaceNet 기준 face image)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FaceDAE(nn.Module):
    """U-Net DAE for 160×160 face denoising.

    Encoder: 160→80→40→20
    Bottleneck: 20×20
    Decoder: 20→40→80→160
    Skip connections: encoder feature maps concat with decoder.
    """

    def __init__(self, base_ch: int = 32) -> None:
        super().__init__()
        # Encoder
        self.enc1 = ConvBnRelu(3, base_ch)           # 160→160
        self.pool1 = nn.MaxPool2d(2)                  # 160→80
        self.enc2 = ConvBnRelu(base_ch, base_ch * 2)  # 80→80
        self.pool2 = nn.MaxPool2d(2)                  # 80→40
        self.enc3 = ConvBnRelu(base_ch * 2, base_ch * 4)  # 40→40
        self.pool3 = nn.MaxPool2d(2)                  # 40→20

        # Bottleneck
        self.bottleneck = ConvBnRelu(base_ch * 4, base_ch * 8)  # 20→20

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBnRelu(base_ch * 8, base_ch * 4)  # skip concat: 4+4=8

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBnRelu(base_ch * 4, base_ch * 2)  # skip concat: 2+2=4

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBnRelu(base_ch * 2, base_ch)      # skip concat: 1+1=2

        # Output: pixel values in [0, 1]
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


def build_dae(base_ch: int = 32) -> FaceDAE:
    return FaceDAE(base_ch=base_ch)


def load_dae(checkpoint_path: str | None, device: torch.device, base_ch: int = 32) -> FaceDAE:
    model = build_dae(base_ch).to(device)
    if checkpoint_path:
        from pathlib import Path
        ckpt_path = Path(checkpoint_path)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state)
            model.eval()
        else:
            raise FileNotFoundError(f"DAE checkpoint not found: {ckpt_path}")
    return model
