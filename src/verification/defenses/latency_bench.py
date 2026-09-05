"""
Detector latency 계측 — PERF-001

`07_DEFENSE_AND_DETECTION_SPEC.md` 7절의 네 번째 잠정 통과 기준은 "명시한 reference
hardware의 latency budget 충족"이다. 앞의 세 기준과 달리 한 번도 측정하지 않았다.

측정 대상은 두 경로다. 둘은 비용 구조가 다르므로 하나의 숫자로 합치지 않는다.

| 경로 | 구현 | 프레임당 변환 | 프레임당 forward |
|---|---|---|---|
| 연구 트랙 계측 | `squeeze_probe.probe_crop` | 14 | 1 (배치 15) |
| face_auth 실시간 게이트 | `feature_squeeze.FeatureSqueezeInspector` | 3 | 3 (배치 1씩) |

실시간 인증 경로는 후자다.

## 계측 규칙

- **워밍업을 제외한다.** 모델 로딩, lazy import, MPS 커널 컴파일이 첫 호출에 섞인다.
  `run_repeats`가 워밍업 반복을 실행하되 표본에서 버린다.
- **p50과 p95를 함께 낸다.** 평균은 꼬리를 감춘다. 백분위는 nearest-rank로 정의한다.
  정의를 적지 않은 p95는 재현되지 않는다.
- **표본 수를 결과에 남긴다.** `09_EVALUATION_METRICS.md` 6절의 분모 명시 원칙이다.
- **단계를 분해한다.** `forward`는 임베더 호출에 실제로 머문 시간이고 `other`는 나머지
  (변환 적용, PIL/numpy 변환, cosine 연산)다. 변환 자체의 비용은 별도의 변환 전용
  시나리오로 교차 확인한다. `other`를 변환 시간이라고 부르지 않는 이유다.

## 한계

- 이 모듈은 시간만 잰다. 탐지 성능과 무관하다.
- 측정값은 실행한 기기 하나의 값이다. `describe_environment()`가 기록하는 항목이
  같지 않으면 다른 기기의 수치와 비교하지 않는다.
- 카메라 캡처, 얼굴 검출(MTCNN), 정책 판정은 포함하지 않는다. detector 게이트만 잰다.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image

from src.face_auth.inference.adversarial_detector import (
    AdversarialDetectorConfig,
    TemplateShiftDetector,
    TemplateShiftDetectorConfig,
    TransformConsistencyDetector,
)
from src.face_auth.inference.feature_squeeze import (
    FeatureSqueezeConfig,
    FeatureSqueezeInspector,
    _transforms as gate_transforms,
)
from src.verification.defenses.randomized_squeeze import (
    sample_transform,
    transform_families,
)
from src.verification.defenses.squeeze_probe import (
    TRANSFORM_ORDER,
    TRANSFORMS,
    probe_crop,
)

# 계측용 임계값. 판정을 쓰지 않으므로 값 자체에 의미가 없다는 것을 이름으로 드러낸다.
BENCH_THRESHOLD_VERSION = "latency-bench-unused"


class EmptySampleError(ValueError):
    """표본이 없는데 백분위를 물었다. 0을 돌려주면 '빠르다'로 읽힌다."""


# ── 집계 ──────────────────────────────────────────────────────────────────────


def percentile_ms(samples: Sequence[float], q: float) -> float:
    """
    nearest-rank 백분위. 정렬한 표본에서 `ceil(q/100 * n)`번째 값을 고른다.

    보간하지 않는다. 보간한 값은 실제로 관측되지 않은 시간이고, 표본이 적을 때
    구현마다 결과가 달라진다.
    """
    if not samples:
        raise EmptySampleError("표본이 비어 있다. 백분위를 계산할 수 없다")
    if not 0 < q <= 100:
        raise ValueError(f"백분위는 (0, 100] 범위여야 한다: {q}")

    ordered = sorted(float(value) for value in samples)
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True)
class LatencySummary:
    label: str
    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


def summarize(label: str, samples: Sequence[float]) -> LatencySummary:
    """표본 목록 하나를 요약한다. 표본 수를 반드시 함께 들고 다닌다."""
    if not samples:
        raise EmptySampleError(f"{label}: 표본이 비어 있다")

    values = [float(value) for value in samples]
    return LatencySummary(
        label=label,
        n=len(values),
        mean_ms=sum(values) / len(values),
        p50_ms=percentile_ms(values, 50),
        p95_ms=percentile_ms(values, 95),
        min_ms=min(values),
        max_ms=max(values),
    )


def summarize_stages(label: str, records: Sequence[dict]) -> dict[str, LatencySummary]:
    """반복마다 얻은 단계별 소요 시간을 단계별 요약으로 바꾼다."""
    if not records:
        raise EmptySampleError(f"{label}: 반복 기록이 비어 있다")

    stages = list(records[0])
    for index, record in enumerate(records):
        if list(record) != stages:
            raise ValueError(
                f"{label}: {index}번째 기록의 단계가 다르다. "
                f"기대 {stages}, 실제 {list(record)}"
            )
    return {
        stage: summarize(f"{label}.{stage}", [record[stage] for record in records])
        for stage in stages
    }


def run_repeats(
    step: Callable[[], dict], repeats: int, warmup: int
) -> list[dict]:
    """
    `warmup + repeats`번 실행하고 뒤의 `repeats`개만 돌려준다.

    워밍업 표본을 남기면 모델 로딩과 커널 컴파일 시간이 p95를 지배한다.
    """
    if repeats <= 0:
        raise ValueError(f"repeats는 1 이상이어야 한다: {repeats}")
    if warmup < 0:
        raise ValueError(f"warmup은 0 이상이어야 한다: {warmup}")

    for _ in range(warmup):
        step()
    return [step() for _ in range(repeats)]


# ── forward 시간 계측 ─────────────────────────────────────────────────────────
#
# 임베더를 감싸서 forward에 머문 시간만 따로 모은다. 운영 코드를 바꾸지 않고
# 실제 호출 경로를 그대로 재는 방법이다.


class ForwardClock:
    def __init__(self) -> None:
        self.elapsed_ms = 0.0

    def reset(self) -> None:
        self.elapsed_ms = 0.0

    def add(self, elapsed_ms: float) -> None:
        self.elapsed_ms += elapsed_ms


class TimedBatchEmbedder:
    """`squeeze_probe.BatchEmbedder` 계약을 감싼다."""

    def __init__(self, inner, clock: ForwardClock) -> None:
        self.inner = inner
        self.clock = clock

    def embed_batch(self, images):
        started = time.perf_counter()
        result = self.inner.embed_batch(images)
        self.clock.add((time.perf_counter() - started) * 1000.0)
        return result


class TimedFrameEmbedder:
    """face_auth 임베더 계약(`embed`)을 감싼다."""

    def __init__(self, inner, clock: ForwardClock) -> None:
        self.inner = inner
        self.clock = clock

    def embed(self, images):
        started = time.perf_counter()
        result = self.inner.embed(images)
        self.clock.add((time.perf_counter() - started) * 1000.0)
        return result


def _timed(clock: ForwardClock, body: Callable[[], None]) -> dict:
    clock.reset()
    started = time.perf_counter()
    body()
    total_ms = (time.perf_counter() - started) * 1000.0
    return {
        "total": total_ms,
        "forward": clock.elapsed_ms,
        "other": total_ms - clock.elapsed_ms,
    }


# ── 입력 ──────────────────────────────────────────────────────────────────────


def synthetic_crop(size: int = 160, seed: int = 0) -> Image.Image:
    """
    합성 크롭. 저주파 성분이 있는 이미지를 만든다.

    균일 잡음을 쓰면 JPEG가 거의 압축하지 못해 실제 얼굴보다 느리게 나온다. 변환
    비용은 내용에 따라 달라지므로 가능하면 `--crop`으로 실제 크롭을 넘긴다.
    """
    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")

    channels = []
    for offset in range(3):
        base = 90.0 + 90.0 * np.sin(2.0 * np.pi * (grid_x + 0.3 * offset)) * np.cos(
            2.0 * np.pi * grid_y
        )
        coarse = rng.normal(0.0, 1.0, (size // 8 + 1, size // 8 + 1)).astype(np.float32)
        upscaled = np.asarray(
            Image.fromarray(coarse, mode="F").resize((size, size), Image.BILINEAR)
        )
        channels.append(base + 18.0 * upscaled)

    stacked = np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)
    return Image.fromarray(stacked, mode="RGB")


def load_crop(path: str | None, size: int, seed: int) -> tuple[Image.Image, str]:
    """실제 크롭이 있으면 그것을, 없으면 합성 크롭을 쓴다. 어느 쪽인지 기록한다."""
    if path is None:
        return synthetic_crop(size, seed), f"synthetic(seed={seed})"
    image = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return image, f"file({Path(path).name})"


# ── 시나리오 ──────────────────────────────────────────────────────────────────


def probe_step(crop, enroll_embedding, embedder) -> Callable[[], dict]:
    """연구 트랙 계측: 변환 14종 + 원본을 한 배치로 forward 1회."""
    clock = ForwardClock()
    timed = TimedBatchEmbedder(embedder, clock)
    enroll = np.asarray(enroll_embedding, dtype=np.float64)

    def step() -> dict:
        return _timed(clock, lambda: probe_crop(crop, enroll, timed))

    return step


def gate_step(
    embedder,
    detector: TransformConsistencyDetector,
    crops: list,
    original_embeddings: list,
    config: FeatureSqueezeConfig,
    template_detector: TemplateShiftDetector | None = None,
) -> Callable[[], dict]:
    """face_auth 게이트: 프레임마다 변환 3종, 최근 `max_frames`프레임."""
    clock = ForwardClock()
    inspector = FeatureSqueezeInspector(
        TimedFrameEmbedder(embedder, clock),
        detector,
        config,
        template_detector=template_detector,
    )

    def step() -> dict:
        return _timed(
            clock, lambda: inspector.evaluate(list(crops), list(original_embeddings))
        )

    return step


def callable_step(fn: Callable[[], object]) -> Callable[[], dict]:
    """
    호출 하나의 소요 시간만 잰다. forward를 변환에서 떼어 내 단독으로 볼 때 쓴다.

    게이트 전체를 재면 배치 크기와 호출 횟수가 섞여서 "forward가 느리다"까지만
    알 수 있다. 같은 이미지 수를 배치 1회와 개별 호출로 나눠 재야 그 느림이 연산
    때문인지 호출 오버헤드 때문인지 구분된다.
    """

    def step() -> dict:
        started = time.perf_counter()
        fn()
        return {"total": (time.perf_counter() - started) * 1000.0}

    return step


def fixed_transform_step(
    crop, names: Sequence[str], sink: Callable[[Image.Image], None] | None = None
) -> Callable[[], dict]:
    """고정 파라미터 변환만 적용한다. forward 없이 변환 비용만 본다."""
    selected = []
    for name in names:
        if name not in TRANSFORMS:
            raise KeyError(f"알 수 없는 변환 {name!r}. 사용 가능: {sorted(TRANSFORMS)}")
        selected.append(TRANSFORMS[name])

    def step() -> dict:
        started = time.perf_counter()
        for transform in selected:
            result = transform(crop)
            if sink is not None:
                sink(result)
        return {"total": (time.perf_counter() - started) * 1000.0}

    return step


def randomized_transform_step(
    crop,
    families: Sequence[str],
    rng,
    sink: Callable | None = None,
) -> Callable[[], dict]:
    """
    계열마다 파라미터를 매번 새로 뽑아 적용한다.

    고정 변환과 비교하려면 적용 횟수가 같아야 한다. 파라미터 추출 시간도 측정에
    포함한다. 그것이 랜덤화가 실제로 추가하는 비용이다.
    """
    known = transform_families()
    for family in families:
        if family not in known:
            raise KeyError(f"알 수 없는 계열 {family!r}. 사용 가능: {sorted(known)}")

    def step() -> dict:
        started = time.perf_counter()
        for family in families:
            spec = sample_transform(family, rng)
            spec.apply(crop)
            if sink is not None:
                sink(spec)
        return {"total": (time.perf_counter() - started) * 1000.0}

    return step


def gate_transform_step(
    crop, config: FeatureSqueezeConfig, sink: Callable | None = None
) -> Callable[[], dict]:
    """face_auth 게이트가 프레임마다 적용하는 변환 3종만 적용한다."""

    def step() -> dict:
        started = time.perf_counter()
        result = gate_transforms(crop, config)
        elapsed = (time.perf_counter() - started) * 1000.0
        if sink is not None:
            sink(result)
        return {"total": elapsed}

    return step


# ── 환경 기록 ─────────────────────────────────────────────────────────────────


def describe_environment(device_name: str) -> dict:
    """
    측정 환경. 이 블록이 없는 latency 수치는 07 7절 기준으로 쓸 수 없다.

    Rosetta 2 여부까지 기록한다. x86_64 빌드가 Apple Silicon에서 번역 실행되면
    CPU 수치가 native arm64보다 느리다.
    """
    import subprocess

    import torch

    def _sysctl(key: str) -> str:
        try:
            return subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, timeout=5
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    translated = _sysctl("sysctl.proc_translated") if sys.platform == "darwin" else ""
    return {
        "device": device_name,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _sysctl("machdep.cpu.brand_string") if sys.platform == "darwin" else "",
        "hardware_model": _sysctl("hw.model") if sys.platform == "darwin" else "",
        "logical_cores": _sysctl("hw.logicalcpu") if sys.platform == "darwin" else "",
        "rosetta_translated": translated,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
#
# 장치는 실행당 하나만 받는다. `facenet_embed.get_model`이 모델을 전역 싱글톤으로
# 들고 있어서 한 프로세스에서 장치를 바꾸면 두 번째 장치가 조용히 무시된다.
# cpu와 가속기를 비교하려면 프로세스를 나눠 두 번 실행한다.


def forward_scenarios(*, crop, gate_images, frame_embedder, batch_embedder):
    """
    forward 격리 시나리오. 같은 이미지 수를 다른 호출 방식으로 잰다.

    meta의 calls와 batch_size는 그 시나리오가 forward를 몇 번, 얼마 크기로 부르는지의
    선언이다. 구현이 바뀌면 선언과 어긋날 수 있으므로 테스트가 스텁 모델로 실제 호출
    수를 세어 대조한다.

    forward_only_single_gate라는 이름이 있었으나 제거했다. 621520a에서
    FaceNetEmbedder.embed가 배치화되면서 그 이름이 가리키는 측정 대상이 바뀌었다.
    이름을 유지하면 배치화 전후 값을 같은 이름으로 비교하게 되는데 서로 다른 것을
    재고 있으므로 성립하지 않는다.
    """

    def loop_gate():
        from src.verification.defenses.facenet_embed import get_embedding

        device = getattr(frame_embedder, "device", None)
        return [get_embedding(crop, device).numpy() for _ in range(gate_images)]

    return {
        "forward_only_single1": (
            callable_step(lambda: frame_embedder.embed([crop])),
            {"calls": 1, "batch_size": 1},
        ),
        "forward_only_loop_gate": (
            callable_step(loop_gate),
            {
                "calls": gate_images,
                "batch_size": 1,
                "role": "게이트 이미지 수만큼 개별 호출. 호출당 고정 비용 기준선",
            },
        ),
        "forward_only_embedder_gate": (
            callable_step(lambda: frame_embedder.embed([crop] * gate_images)),
            {
                "calls": 1,
                "batch_size": gate_images,
                "role": "현재 게이트 경로. 621520a 이후 배치 1회다",
            },
        ),
        "forward_only_batch_gate": (
            callable_step(lambda: batch_embedder.embed_batch([crop] * gate_images)),
            {
                "calls": 1,
                "batch_size": gate_images,
                "role": "연구 트랙 embed_batch. embedder_gate와 같은 값이어야 한다",
            },
        ),
    }


def _build_report(args) -> dict:
    from src.verification.defenses.facenet_embed import (
        FaceNetBatchEmbedder,
        get_embedding,
        get_model,
        select_device,
    )
    from src.face_auth.inference.verifier import FaceNetEmbedder

    device = select_device(args.device)
    model, resolved = get_model(device)
    if resolved.type != device.type:
        raise RuntimeError(
            f"요청한 장치 {device}가 아니라 {resolved}에 모델이 올라갔다. "
            "프로세스를 새로 띄워 한 번에 한 장치만 측정한다"
        )

    crop, crop_source = load_crop(args.crop, args.crop_size, args.seed)
    config = FeatureSqueezeConfig(max_frames=args.max_frames)

    batch_embedder = FaceNetBatchEmbedder(resolved)
    frame_embedder = FaceNetEmbedder(resolved)
    enroll = get_embedding(crop, resolved).numpy().astype(np.float64)
    original = enroll.copy()

    detector = TransformConsistencyDetector(
        AdversarialDetectorConfig(
            max_cosine_distance=1.0, threshold_version=BENCH_THRESHOLD_VERSION
        )
    )
    template_detector = TemplateShiftDetector(
        TemplateShiftDetectorConfig(
            max_template_shift=1.0, threshold_version=BENCH_THRESHOLD_VERSION
        ),
        enroll,
    )
    frames = [crop] * args.max_frames
    originals = [original] * args.max_frames
    # 게이트가 한 번 판정하는 동안 실제로 임베딩하는 이미지 수 (변환 3종 x 프레임 수)
    gate_images = 3 * args.max_frames
    _params_rng = np.random.default_rng(args.seed)

    scenarios: dict[str, tuple[Callable[[], dict], dict]] = {
        "research_probe": (
            probe_step(crop, enroll, batch_embedder),
            {"transforms_per_frame": len(TRANSFORM_ORDER), "batch_size": len(TRANSFORM_ORDER) + 1},
        ),
        "face_auth_gate_1": (
            gate_step(frame_embedder, detector, frames, originals, config),
            {"gates": 1, "frames": args.max_frames, "transforms_per_frame": 3},
        ),
        "face_auth_gate_2": (
            gate_step(
                frame_embedder,
                detector,
                frames,
                originals,
                config,
                template_detector=template_detector,
            ),
            {"gates": 2, "frames": args.max_frames, "transforms_per_frame": 3},
        ),
        "transform_only_research": (
            fixed_transform_step(crop, TRANSFORM_ORDER),
            {"transforms": len(TRANSFORM_ORDER)},
        ),
        "transform_only_gate": (
            gate_transform_step(crop, config),
            {"transforms": 3},
        ),
        "transform_only_fixed3": (
            fixed_transform_step(crop, ("blur0.8", "jpeg_q75", "median3")),
            {"transforms": 3, "role": "randomized 비교 기준선"},
        ),
        "transform_only_randomized3": (
            randomized_transform_step(
                crop, ("blur", "jpeg", "median"), np.random.default_rng(args.seed)
            ),
            {"transforms": 3, "role": "계열마다 파라미터 재추출"},
        ),
        # 파라미터 추출만. randomized와 fixed의 차이 중 어디까지가 랜덤화 자체의
        # 비용이고 어디부터가 파라미터 분포 차이인지 가른다.
        "transform_only_sample_params3": (
            callable_step(
                lambda: [
                    sample_transform(family, _params_rng)
                    for family in ("blur", "jpeg", "median")
                ]
            ),
            {"draws": 3, "role": "적용 없이 추출만"},
        ),
    }
    # forward 격리. 같은 이미지 수를 다른 호출 방식으로 재면 느림이 연산 때문인지
    # 호출당 고정 비용 때문인지 갈린다.
    scenarios.update(
        forward_scenarios(
            crop=crop,
            gate_images=gate_images,
            frame_embedder=frame_embedder,
            batch_embedder=batch_embedder,
        )
    )

    results = {}
    for name, (step, meta) in scenarios.items():
        records = run_repeats(step, repeats=args.repeats, warmup=args.warmup)
        results[name] = {
            "meta": meta,
            "stages": {
                stage: summary.as_dict()
                for stage, summary in summarize_stages(name, records).items()
            },
        }

    return {
        "requirement_id": "PERF-001",
        "environment": describe_environment(str(resolved)),
        "measurement": {
            "repeats": args.repeats,
            "warmup": args.warmup,
            "crop_size": args.crop_size,
            "crop_source": crop_source,
            "max_frames": args.max_frames,
            "seed": args.seed,
            "model_version": FaceNetEmbedder.model_version,
            "percentile_method": "nearest-rank",
        },
        "scenarios": results,
        "limitations": [
            "단일 기기 단일 실행이다. 기기가 다르면 이 수치를 재사용하지 않는다.",
            "카메라 캡처, MTCNN 얼굴 검출, 정책 판정은 포함하지 않는다. detector 게이트만 잰다.",
            "07 7절이 요구하는 latency budget 수치가 저장소에 정의돼 있지 않다. "
            "이 실행은 예산 충족 여부를 판정하지 않고 측정값만 제공한다.",
            "same-crop 반복이므로 프레임마다 내용이 달라지는 실제 세션의 분산을 반영하지 않는다.",
        ],
    }


def _print_report(report: dict) -> None:
    environment = report["environment"]
    print(f"장치 {environment['device']}  torch {environment['torch']}  "
          f"threads {environment['torch_threads']}")
    print(f"{environment['cpu_model'] or environment['machine']}  {environment['platform']}")
    measurement = report["measurement"]
    print(f"표본 {measurement['repeats']}회 (워밍업 {measurement['warmup']}회 제외), "
          f"크롭 {measurement['crop_size']}px {measurement['crop_source']}")
    print()
    header = f"{'시나리오':<28} {'단계':<9} {'n':>4} {'p50(ms)':>9} {'p95(ms)':>9} {'mean':>9}"
    print(header)
    print("-" * len(header))
    for name, payload in report["scenarios"].items():
        for stage, summary in payload["stages"].items():
            print(f"{name:<28} {stage:<9} {summary['n']:>4} "
                  f"{summary['p50_ms']:>9.2f} {summary['p95_ms']:>9.2f} "
                  f"{summary['mean_ms']:>9.2f}")
    print()
    print(f"한계 {len(report['limitations'])}건:")
    for item in report["limitations"]:
        print(f"  - {item}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PERF-001 detector latency 측정. 한 번에 한 장치만 측정한다"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cpu, mps, cuda 중 하나. 생략하면 select_device()가 고른다",
    )
    parser.add_argument("--repeats", type=int, default=50, help="측정 반복 수")
    parser.add_argument("--warmup", type=int, default=10, help="버릴 워밍업 반복 수")
    parser.add_argument("--crop", default=None, help="실제 얼굴 크롭 경로. 없으면 합성")
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument(
        "--max-frames", type=int, default=3, help="face_auth 게이트의 윈도 크기"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="JSON 저장 경로")
    args = parser.parse_args()

    report = _build_report(args)
    _print_report(report)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
