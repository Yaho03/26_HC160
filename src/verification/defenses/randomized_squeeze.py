"""
랜덤화 squeeze 변환

BPDA가 통하는 이유는 공격 루프에서 방어의 변환을 그대로 재현할 수 있기 때문이다.
파라미터를 매번 무작위로 뽑으면 공격자가 고정 목표를 잡지 못한다.

07_DEFENSE_AND_DETECTION_SPEC.md 4절에 따라 이것을 stochastic heuristic으로 다룬다.
certificate를 만들지 않으므로 보장이 아니라 경험적 방어다. 랜덤화를 아는 공격자는
분포 전체에 대해 EOT를 걸 수 있으며, 그 평가 없이 내성을 주장하지 않는다.

파라미터 범위는 고정 변환 실측에서 유용했던 구간을 중심으로 잡았다. 범위를 넓히면
공격자가 맞히기 어려워지지만 clean 표본의 분산도 함께 커진다.

bit 계열은 face_auth 게이트가 쓰는 계열(jpeg, bit, blur)을 그대로 랜덤화할 수 있게
두었다. 랜덤화와 계열 교체를 한 번에 하면 탐지율 변화의 원인을 가릴 수 없다.
연구 트랙 웹캠 실측에서 비트깊이는 약했으나(self_consistency AUC 0.42,
template_shift 0.30) face_auth 경로에서는 측정한 적이 없다.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageFilter

# family -> (최소, 최대).
#
# narrow는 고정 변환 실측에서 분리도가 있던 구간을 덮는다. wide는 공격자가 맞히기
# 어렵게 넓힌 것이다. 범위를 넓히면 clean 점수의 분산도 함께 커지므로 어느 쪽이
# 이기는지는 측정해야 한다.
RANGE_PRESETS = {
    "narrow": {
        "blur": (0.5, 2.0),
        "jpeg": (30, 75),
        "median": (3, 5),
        "bit": (4, 6),
    },
    "wide": {
        "blur": (0.3, 3.5),
        "jpeg": (10, 90),
        "median": (3, 7),
        "bit": (3, 7),
    },
}

_FAMILIES = RANGE_PRESETS["narrow"]


class UnknownFamilyError(ValueError):
    """선언되지 않은 변환 계열."""


def transform_families() -> dict[str, tuple]:
    return dict(_FAMILIES)


@dataclass(frozen=True)
class RandomizedTransformSpec:
    family: str
    params: dict

    def apply(self, image: Image.Image) -> Image.Image:
        if self.family == "blur":
            return image.convert("RGB").filter(
                ImageFilter.GaussianBlur(self.params["radius"])
            )
        if self.family == "jpeg":
            buffer = BytesIO()
            image.convert("RGB").save(
                buffer, format="JPEG", quality=int(self.params["quality"])
            )
            buffer.seek(0)
            return Image.open(buffer).convert("RGB").copy()
        if self.family == "median":
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            return Image.fromarray(cv2.medianBlur(array, int(self.params["kernel"])))
        if self.family == "bit":
            # face_auth feature_squeeze.py 와 같은 반올림 양자화
            levels = float((1 << int(self.params["bits"])) - 1)
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            quantized = np.rint(array * levels) / levels * 255.0
            return Image.fromarray(np.clip(quantized, 0, 255).astype(np.uint8))
        raise UnknownFamilyError(self.family)


def sample_transform(family: str, rng, preset: str = "narrow") -> RandomizedTransformSpec:
    """
    계열에서 파라미터를 하나 뽑는다. rng를 받아 재현 가능하게 한다.

    median kernel은 홀수여야 하므로 후보 중에서 고른다.
    """
    ranges = RANGE_PRESETS[preset]
    if family not in ranges:
        raise UnknownFamilyError(
            f"알 수 없는 계열 {family!r}. 사용 가능: {sorted(ranges)}"
        )
    low, high = ranges[family]

    if family == "blur":
        return RandomizedTransformSpec(family, {"radius": float(rng.uniform(low, high))})
    if family == "jpeg":
        return RandomizedTransformSpec(
            family, {"quality": int(rng.integers(low, high + 1))}
        )
    if family == "bit":
        return RandomizedTransformSpec(family, {"bits": int(rng.integers(low, high + 1))})
    kernels = [k for k in (3, 5, 7) if low <= k <= high]
    return RandomizedTransformSpec(family, {"kernel": int(rng.choice(kernels))})
