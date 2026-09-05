"""
BPDA — Backward Pass Differentiable Approximation

EOT는 미분 가능한 변환만 공격 루프에 넣을 수 있다. detector가 쓰는 특징의 대부분이
JPEG와 median filter에서 나오므로, EOT만으로는 점수의 대부분을 건드리지 못한다.

BPDA는 forward에 진짜 변환을 쓰고 backward만 항등함수로 근사한다.

    y = x + (T(x).detach() - x.detach())

forward에서 y는 정확히 T(x)이고, backward에서 dy/dx는 항등이다. 근사가 backward에만
적용되므로 공격자가 보는 detector 출력은 방어가 실제로 계산하는 값과 같다.

이 모듈은 방어의 강건성이 미분 불가능성에 의존하는지 확인하기 위한 것이다.
07_DEFENSE_AND_DETECTION_SPEC.md 7.1절이 이 우회를 명시적으로 지목한다.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.verification.defenses.squeeze_probe import TRANSFORMS

# blur와 lowres는 미분 가능하므로 근사가 필요 없다. adaptive_attack이 직접 다룬다.
_NON_DIFFERENTIABLE = ("jpeg_q75", "jpeg_q50", "jpeg_q30", "jpeg_q10", "median3", "median5", "median7")


def supported_transforms() -> tuple[str, ...]:
    """BPDA가 필요한 변환. 미분 가능한 것은 여기 없다."""
    return tuple(name for name in _NON_DIFFERENTIABLE if name in TRANSFORMS)


def _to_pil(tensor: torch.Tensor) -> Image.Image:
    """정규화 공간 (1,3,H,W) → PIL. 실제 파이프라인과 같은 uint8 왕복을 거친다."""
    array = tensor.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    array = (array * 128.0 + 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


def _to_tensor(image: Image.Image, device) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    array = (array - 127.5) / 128.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def bpda_transform(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """
    forward는 실제 변환, backward는 항등 근사.

    detach된 차이를 더하므로 값은 T(x)와 같고 gradient는 x로 그대로 흐른다.
    """
    if name not in TRANSFORMS:
        raise KeyError(f"알 수 없는 변환 {name!r}. 사용 가능: {sorted(TRANSFORMS)}")

    transformed = _to_tensor(TRANSFORMS[name](_to_pil(tensor)), tensor.device)
    return tensor + (transformed.detach() - tensor.detach())


def bpda_transform_batch(tensor: torch.Tensor, names) -> torch.Tensor:
    """
    변환 여러 개를 한 배치로 묶는다. 값은 개별 호출과 같고 forward 횟수만 줄어든다.

    공격 루프는 스텝마다 변환 전부를 통과시키므로, 묶지 않으면 forward가 변환 수만큼
    늘어난다. 표본이 늘면 그 차이가 그대로 실행 시간이 된다.
    """
    names = list(names)
    if not names:
        raise ValueError("변환을 하나 이상 지정해야 한다")

    unknown = [name for name in names if name not in TRANSFORMS]
    if unknown:
        raise KeyError(f"알 수 없는 변환 {unknown}. 사용 가능: {sorted(TRANSFORMS)}")

    source = _to_pil(tensor)
    transformed = torch.cat(
        [_to_tensor(TRANSFORMS[name](source), tensor.device) for name in names]
    )
    base = tensor.expand(len(names), -1, -1, -1)
    return base + (transformed.detach() - base.detach())


def bpda_spec_batch(tensor: torch.Tensor, specs) -> torch.Tensor:
    """
    RandomizedTransformSpec 목록에 대한 BPDA 배치.

    랜덤화를 아는 공격자는 매 스텝 분포에서 새로 뽑아 EOT를 건다. 고정 파라미터로
    공격하면 방어가 실제로 쓸 파라미터와 어긋나 공격이 약해진다. 그 차이를 재려면
    두 공격자를 모두 평가해야 한다.
    """
    specs = list(specs)
    if not specs:
        raise ValueError("변환 spec을 하나 이상 지정해야 한다")

    source = _to_pil(tensor)
    transformed = torch.cat(
        [_to_tensor(spec.apply(source), tensor.device) for spec in specs]
    )
    base = tensor.expand(len(specs), -1, -1, -1)
    return base + (transformed.detach() - base.detach())
