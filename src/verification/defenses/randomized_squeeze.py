"""
랜덤화 squeeze 변환

BPDA가 통하는 이유는 공격 루프에서 방어의 변환을 그대로 재현할 수 있기 때문이다.
파라미터를 매번 무작위로 뽑으면 공격자가 고정 목표를 잡지 못한다.

07_DEFENSE_AND_DETECTION_SPEC.md 4절에 따라 이것을 stochastic heuristic으로 다룬다.
certificate를 만들지 않으므로 보장이 아니라 경험적 방어다. 랜덤화를 아는 공격자는
분포 전체에 대해 EOT를 걸 수 있으며, 그 평가 없이 내성을 주장하지 않는다.

파라미터 범위는 고정 변환 실측에서 유용했던 구간을 중심으로 잡았다. 범위를 넓히면
공격자가 맞히기 어려워지지만 clean 표본의 분산도 함께 커진다.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageFilter

# family -> (최소, 최대). 실측에서 분리도가 있던 구간을 덮는다.
_FAMILIES = {
    "blur": (0.5, 2.0),
    "jpeg": (30, 75),
    "median": (3, 5),
}


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
        raise UnknownFamilyError(self.family)


def sample_transform(family: str, rng) -> RandomizedTransformSpec:
    """
    계열에서 파라미터를 하나 뽑는다. rng를 받아 재현 가능하게 한다.

    median kernel은 홀수여야 하므로 후보 중에서 고른다.
    """
    if family not in _FAMILIES:
        raise UnknownFamilyError(
            f"알 수 없는 계열 {family!r}. 사용 가능: {sorted(_FAMILIES)}"
        )
    low, high = _FAMILIES[family]

    if family == "blur":
        return RandomizedTransformSpec(family, {"radius": float(rng.uniform(low, high))})
    if family == "jpeg":
        return RandomizedTransformSpec(
            family, {"quality": int(rng.integers(low, high + 1))}
        )
    kernels = [k for k in (3, 5) if low <= k <= high]
    return RandomizedTransformSpec(family, {"kernel": int(rng.choice(kernels))})
