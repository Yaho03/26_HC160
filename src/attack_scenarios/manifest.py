from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InsertVideoSpec:
    at_index: int
    path: Path
    max_frames: int | None = None


@dataclass(frozen=True)
class RepeatFrameSpec:
    source_index: int
    at_index: int
    count: int


@dataclass(frozen=True)
class ScenarioManifest:
    scenario_id: str
    base_video: Path
    output_video: Path
    fps: float
    events: tuple[InsertVideoSpec | RepeatFrameSpec, ...]


def load_manifest(path: str | Path) -> ScenarioManifest:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    scenario_id = _required_text(payload, "scenario_id")
    fps = float(payload.get("fps", 20.0))
    if fps <= 0:
        raise ValueError("fps must be positive")
    events = tuple(_event(item, root) for item in payload.get("events", []))
    if not events:
        raise ValueError("At least one scenario event is required")
    return ScenarioManifest(
        scenario_id=scenario_id,
        base_video=_relative_path(root, _required_text(payload, "base_video")),
        output_video=_relative_path(root, _required_text(payload, "output_video")),
        fps=fps,
        events=events,
    )


def _event(payload: dict[str, Any], root: Path):
    kind = payload.get("type")
    if kind == "insert_video":
        at_index = _nonnegative_int(payload, "at_index")
        max_frames = payload.get("max_frames")
        if max_frames is not None and int(max_frames) <= 0:
            raise ValueError("max_frames must be positive")
        return InsertVideoSpec(
            at_index=at_index,
            path=_relative_path(root, _required_text(payload, "path")),
            max_frames=int(max_frames) if max_frames is not None else None,
        )
    if kind == "repeat_frame":
        count = int(payload.get("count", 0))
        if count <= 0:
            raise ValueError("repeat_frame count must be positive")
        return RepeatFrameSpec(
            source_index=_nonnegative_int(payload, "source_index"),
            at_index=_nonnegative_int(payload, "at_index"),
            count=count,
        )
    raise ValueError(f"Unsupported scenario event type: {kind}")


def _relative_path(root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError("Scenario paths must be relative to the manifest")
    return (root / candidate).resolve()


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _nonnegative_int(payload: dict[str, Any], key: str) -> int:
    value = int(payload.get(key, -1))
    if value < 0:
        raise ValueError(f"{key} must be non-negative")
    return value
