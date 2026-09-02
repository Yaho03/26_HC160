"""
Probe log — 계측 결과 기록

CSV 행과 세션 사이드카를 쓴다. 스키마를 고정하고, 얼굴 원본·임베딩·절대 경로가
산출물에 섞이는 것을 거부한다.

CSV는 행 단위 형식이다. 표본 하나가 변환 6종에 대해 6행을 만든다. 변환을 추가하거나
제거해도 스키마가 바뀌지 않고, 변환별·측정량별 ROC를 pivot 한 번으로 뽑을 수 있다.

설계 근거는 docs/experiments/EXP-DET-001-camera-squeeze-probe.md 참조.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from src.verification.defenses.squeeze_probe import ProbeReading

PROBE_COLUMNS: tuple[str, ...] = (
    "session_id",
    "subject_id",
    "sample_id",
    "frame_idx",
    "frame_ts_ms",
    "dropped_frames",
    "label",
    "transform",
    "cos_orig_enroll",
    "cos_transformed_enroll",
    "cos_orig_transformed",
    "embed_ms",
)

LABELS: frozenset[str] = frozenset({"clean", "adversarial"})

# 불투명 ID만 허용한다. 실명, 이메일, 파일 경로가 들어오면 거부한다.
_OPAQUE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# 사이드카에 넣으면 안 되는 키. 임베딩 벡터와 얼굴 원본이 여기로 새는 것을 막는다.
_FORBIDDEN_KEY = re.compile(
    r"embedding|descriptor|crop|image|photo|frame_data|template", re.IGNORECASE
)
_ABSOLUTE_PATH = re.compile(r"^(/|[A-Za-z]:[\\/]|~/)")


class OpaqueIdError(ValueError):
    """세션 또는 피험자 ID가 불투명 ID 규칙을 어겼다."""


class SidecarContentError(ValueError):
    """사이드카에 절대 경로나 생체 원본이 들어 있다."""


def _require_opaque(name: str, value: str) -> str:
    if not isinstance(value, str) or not _OPAQUE_ID.match(value):
        raise OpaqueIdError(
            f"{name}는 영문자·숫자·하이픈·밑줄 64자 이내의 불투명 ID여야 한다: {value!r}"
        )
    return value


class ProbeWriter:
    """계측 CSV를 연다. 헤더는 파일이 비어 있을 때만 쓴다."""

    def __init__(self, path, *, session_id: str, subject_id: str) -> None:
        self.session_id = _require_opaque("session_id", session_id)
        self.subject_id = _require_opaque("subject_id", subject_id)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        write_header = not self.path.exists() or self.path.stat().st_size == 0
        self._handle = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=PROBE_COLUMNS)
        if write_header:
            self._writer.writeheader()
            self._handle.flush()

    def write_sample(
        self,
        *,
        sample_id: str,
        frame_idx: int,
        frame_ts_ms: float,
        dropped_frames: int,
        label: str,
        reading: ProbeReading,
    ) -> int:
        """표본 하나를 변환 수만큼의 행으로 기록하고 기록한 행 수를 반환한다."""
        if label not in LABELS:
            raise ValueError(f"label은 {sorted(LABELS)} 중 하나여야 한다: {label!r}")

        for item in reading.readings:
            self._writer.writerow(
                {
                    "session_id": self.session_id,
                    "subject_id": self.subject_id,
                    "sample_id": sample_id,
                    "frame_idx": frame_idx,
                    "frame_ts_ms": round(frame_ts_ms, 3),
                    "dropped_frames": dropped_frames,
                    "label": label,
                    "transform": item.transform,
                    "cos_orig_enroll": round(item.cos_orig_enroll, 8),
                    "cos_transformed_enroll": round(item.cos_transformed_enroll, 8),
                    "cos_orig_transformed": round(item.cos_orig_transformed, 8),
                    "embed_ms": round(reading.embed_ms, 3),
                }
            )
        self._handle.flush()
        return len(reading.readings)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "ProbeWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _reject_sensitive(node, trail: str = "") -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            where = f"{trail}.{key}" if trail else str(key)
            if _FORBIDDEN_KEY.search(str(key)):
                raise SidecarContentError(f"사이드카에 생체 원본을 넣을 수 없다: {where}")
            _reject_sensitive(value, where)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _reject_sensitive(value, f"{trail}[{index}]")
    elif isinstance(node, str) and _ABSOLUTE_PATH.match(node):
        raise SidecarContentError(f"사이드카에 절대 경로를 넣을 수 없다: {trail}")


def write_session_sidecar(path, meta: dict) -> Path:
    """세션 provenance를 JSON으로 남긴다. 임계값 artifact가 이 값을 참조한다."""
    _reject_sensitive(meta)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target
