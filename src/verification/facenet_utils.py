"""Utilities for FaceNet-style verification with facenet-pytorch."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.common.device import get_device

FACENET_IMAGE_SIZE = (160, 160)


def require_facenet():
    try:
        from facenet_pytorch import InceptionResnetV1
    except ImportError as exc:
        raise ImportError(
            "facenet-pytorch is required. In Colab, run: "
            "pip install -q facenet-pytorch"
        ) from exc
    return InceptionResnetV1


def build_facenet_model(pretrained: str) -> tuple[torch.nn.Module, torch.device]:
    InceptionResnetV1 = require_facenet()
    device = get_device()
    model = InceptionResnetV1(pretrained=pretrained).eval().to(device)
    return model, device


def facenet_pixel_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(FACENET_IMAGE_SIZE),
        transforms.ToTensor(),
    ])


def facenet_model_input(pixel_tensor: torch.Tensor) -> torch.Tensor:
    return (pixel_tensor - 0.5) / 0.5


def load_facenet_image(path: Path, to_pixel_tensor, device: torch.device) -> torch.Tensor:
    return to_pixel_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def facenet_embedding(model: torch.nn.Module, pixel_tensor: torch.Tensor) -> torch.Tensor:
    features = model(facenet_model_input(pixel_tensor))
    return F.normalize(features, p=2, dim=1)


def cosine_score(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left, right).detach().cpu().item())
