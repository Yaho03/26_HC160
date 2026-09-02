"""
Squeeze probe — 계측 전용 순수 계산 모듈

얼굴 크롭 하나에 squeezing 변환 6종을 적용하고, 변환마다 원시 cosine 값 세 개를
계산한다. 임계값 판정은 하지 않는다. 캘리브레이션 데이터를 모으는 단계에서
임계값을 미리 적용하면 측정하려는 대상을 입력으로 되먹이게 된다.

원시값 세 개를 저장하는 이유는 두 detector의 측정량이 모두 여기서 재계산되기
때문이다. 측정 정의를 바꿔도 재촬영할 필요가 없다.

    self_consistency = 1 − cos(원본, 변환)              face_auth 게이트
    template_shift   = |cos(원본,등록) − cos(변환,등록)|  연구 트랙 게이트

설계 근거는 docs/experiments/EXP-DET-001-camera-squeeze-probe.md 참조.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from io import BytesIO
from typing import Callable, Protocol, Sequence

import cv2
import numpy as np
from PIL import Image, ImageFilter


# ── 변환 세트 ─────────────────────────────────────────────────────────────────
#
# 변환은 파라미터가 붙은 팩토리로 만든다. 이름이 곧 파라미터를 읽게 한다.
#
# 세트 구성 근거는 LFW 공격 패키지 120쌍 스윕이다. 상세와 한계는
# docs/experiments/EXP-DET-001-camera-squeeze-probe.md 9절을 참조한다.
# 그 결과는 정지 이미지에서 얻은 것이므로 웹캠 순위를 확정하지 못한다. 따라서
# 승자만 남기지 않고 탐색용과 기준선을 함께 기록한다.


def jpeg(quality: int) -> Callable[[Image.Image], Image.Image]:
    """JPEG 재압축. 원본이 이미 JPEG이면 높은 quality는 거의 무손실이라 약하다."""

    def transform(image: Image.Image) -> Image.Image:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB").copy()

    return transform


def bit_round(bits: int) -> Callable[[Image.Image], Image.Image]:
    """비트깊이 축소, 반올림 양자화. face_auth feature_squeeze.py 와 같은 공식."""

    def transform(image: Image.Image) -> Image.Image:
        levels = float((1 << bits) - 1)
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        quantized = np.rint(array * levels) / levels * 255.0
        return Image.fromarray(np.clip(quantized, 0, 255).astype(np.uint8))

    return transform


def bit_floor(bits: int) -> Callable[[Image.Image], Image.Image]:
    """비트깊이 축소, 내림 양자화. 연구 트랙 squeeze_color_depth 와 같은 공식."""

    def transform(image: Image.Image) -> Image.Image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        levels = 2 ** bits
        quantized = np.floor(array / 256.0 * levels) / levels * 256.0
        return Image.fromarray(np.clip(quantized, 0, 255).astype(np.uint8))

    return transform


def blur(radius: float) -> Callable[[Image.Image], Image.Image]:
    """Gaussian blur. 정지 이미지 스윕에서 가장 강했다."""

    def transform(image: Image.Image) -> Image.Image:
        return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius))

    return transform


def lowres(size: int) -> Callable[[Image.Image], Image.Image]:
    """축소 후 복원. perturbation과 함께 얼굴 구조도 지우므로 약하다."""

    def transform(image: Image.Image) -> Image.Image:
        original_size = image.size
        small = image.convert("RGB").resize((size, size), Image.BILINEAR)
        return small.resize(original_size, Image.BILINEAR)

    return transform


def median(kernel: int) -> Callable[[Image.Image], Image.Image]:
    """Median filter. 국소 perturbation 평활화."""

    def transform(image: Image.Image) -> Image.Image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return Image.fromarray(cv2.medianBlur(array, kernel))

    return transform


# 스윕 상위권. 두 측정량 모두에서 AUC 0.99 이상이었다.
_CORE = {
    "blur0.5": (blur(0.5), {"radius": 0.5}),
    "blur0.8": (blur(0.8), {"radius": 0.8}),
    "blur1.2": (blur(1.2), {"radius": 1.2}),
    "median3": (median(3), {"kernel": 3}),
    "median5": (median(5), {"kernel": 5}),
    "jpeg_q30": (jpeg(30), {"quality": 30}),
}

# 웹캠에서 순위가 뒤집힐 경우를 대비한 탐색용. 정지 이미지에서는 중위권이었다.
_EXPLORATORY = {
    "blur2.0": (blur(2.0), {"radius": 2.0}),
    "median7": (median(7), {"kernel": 7}),
    "jpeg_q50": (jpeg(50), {"quality": 50}),
    "jpeg_q10": (jpeg(10), {"quality": 10}),
    "lowres64": (lowres(64), {"low_res": 64, "resample": "bilinear"}),
    "bit4_floor": (bit_floor(4), {"bits": 4, "quantizer": "floor"}),
}

# 현재 배포된 설정. 새 세트와의 차이를 같은 조건에서 재기 위한 기준선이다.
_BASELINE = {
    "jpeg_q75": (jpeg(75), {"quality": 75, "role": "baseline"}),
    "bit5_round": (bit_round(5), {"bits": 5, "quantizer": "round", "role": "baseline"}),
}

_ALL = {**_CORE, **_EXPLORATORY, **_BASELINE}

TRANSFORMS: dict[str, Callable[[Image.Image], Image.Image]] = {
    name: pair[0] for name, pair in _ALL.items()
}
TRANSFORM_PARAMS: dict[str, dict] = {name: pair[1] for name, pair in _ALL.items()}
TRANSFORM_ORDER: tuple[str, ...] = tuple(_ALL)

CORE_TRANSFORMS: tuple[str, ...] = tuple(_CORE)
BASELINE_TRANSFORMS: tuple[str, ...] = tuple(_BASELINE)


# ── 임베더 계약 ───────────────────────────────────────────────────────────────


class BatchEmbedder(Protocol):
    def embed_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        """이미지 n장 → (n, d) L2 정규화 임베딩 행렬."""


# ── 측정 결과 ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TransformReading:
    transform: str
    cos_orig_enroll: float
    cos_transformed_enroll: float
    cos_orig_transformed: float


@dataclass(frozen=True)
class ProbeReading:
    readings: tuple[TransformReading, ...]
    embed_ms: float


def probe_crop(
    crop: Image.Image,
    enroll_embedding: np.ndarray,
    embedder: BatchEmbedder,
) -> ProbeReading:
    """
    크롭 하나에 변환 6종을 적용하고 변환별 원시 cosine 값을 계산한다.

    원본과 변환본을 하나의 배치로 묶어 임베딩을 1회에 계산한다. 변환마다 따로
    호출하면 실시간 기록이 불가능하다.
    """
    images = [crop] + [TRANSFORMS[name](crop) for name in TRANSFORM_ORDER]

    started = time.perf_counter()
    embeddings = np.asarray(embedder.embed_batch(images), dtype=np.float64)
    embed_ms = (time.perf_counter() - started) * 1000.0

    enroll = np.asarray(enroll_embedding, dtype=np.float64)
    original = embeddings[0]
    cos_orig_enroll = float(np.dot(original, enroll))

    readings = tuple(
        TransformReading(
            transform=name,
            cos_orig_enroll=cos_orig_enroll,
            cos_transformed_enroll=float(np.dot(embeddings[index + 1], enroll)),
            cos_orig_transformed=float(np.dot(original, embeddings[index + 1])),
        )
        for index, name in enumerate(TRANSFORM_ORDER)
    )
    return ProbeReading(readings=readings, embed_ms=embed_ms)


# ── 파생 측정량 ───────────────────────────────────────────────────────────────
#
# 저장한 원시값만으로 두 게이트가 재구성된다는 것을 코드로 고정해 둔다.


def jpeg_headroom(crop: Image.Image, quality: int = 75) -> float:
    """
    JPEG 재압축이 실제로 픽셀을 얼마나 바꾸는지. 0~255 척도의 평균 절대 변화량.

    입력이 이미 같은 수준으로 JPEG 압축돼 있으면 재압축이 거의 무손실이라 이 값이
    0에 가까워지고, JPEG 계열 변환은 탐지 신호를 만들지 못한다. macOS AVFoundation은
    CAP_PROP_FOURCC를 보고하지 않으므로 코덱 대신 이 값으로 판단한다.

    참고값: LFW 정지 JPEG 0.076, 웹캠 160x160 크롭 3.21.
    """
    original = np.asarray(crop.convert("RGB"), dtype=np.float64)
    compressed = np.asarray(jpeg(quality)(crop), dtype=np.float64)
    return float(np.mean(np.abs(original - compressed)))


def self_consistency(reading: TransformReading) -> float:
    """face_auth 게이트 측정량. 등록 템플릿과 무관하게 조작 흔적을 본다."""
    return 1.0 - reading.cos_orig_transformed


def template_shift(reading: TransformReading) -> float:
    """연구 트랙 게이트 측정량. 등록자로 위장하는 방향인지를 본다."""
    return abs(reading.cos_orig_enroll - reading.cos_transformed_enroll)
