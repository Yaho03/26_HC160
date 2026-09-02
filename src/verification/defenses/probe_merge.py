"""
계측 CSV 합치기

세션마다 컬럼이 다를 수 있다. `attack_kind`는 나중에 추가된 컬럼이라 그 이전 세션에는
없다. 기존 산출물에 스키마를 소급 적용하지 않는다는 원칙에 따라 원본은 그대로 두고,
합칠 때만 정규화한다.

adversarial 행에 공격 종류가 없으면 `unspecified`로 채운다. 비어 있게 두면 나중에
공격 종류별 보고에서 clean과 구분되지 않는다.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.verification.defenses.probe_log import PROBE_COLUMNS


class DuplicateSessionError(ValueError):
    """같은 세션이 두 번 들어왔다. 표본이 부풀려진다."""


def merge_probe_csvs(sources, destination) -> Path:
    sources = [Path(path) for path in sources]
    if not sources:
        raise ValueError("합칠 CSV를 하나 이상 지정해야 한다")

    merged: list[dict] = []
    seen_sessions: set[str] = set()

    for path in sources:
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        sessions = {row["session_id"] for row in rows}
        overlap = sessions & seen_sessions
        if overlap:
            raise DuplicateSessionError(
                f"세션 {sorted(overlap)}이 이미 포함돼 있다: {path}"
            )
        seen_sessions |= sessions

        for row in rows:
            normalised = {column: row.get(column, "") for column in PROBE_COLUMNS}
            if not normalised["attack_kind"] and row.get("label") == "adversarial":
                normalised["attack_kind"] = "unspecified"
            merged.append(normalised)

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROBE_COLUMNS)
        writer.writeheader()
        writer.writerows(merged)
    return destination
