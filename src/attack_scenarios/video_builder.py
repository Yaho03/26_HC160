from __future__ import annotations

from pathlib import Path

import cv2

from src.attack_scenarios.manifest import (
    InsertVideoSpec,
    RepeatFrameSpec,
    ScenarioManifest,
)
from src.attack_scenarios.scenario_runner import InsertFrames, RepeatFrame, apply_events
from src.face_auth.adapters.opencv_capture import video_source
from src.face_auth.domain.types import FramePacket


def build_scenario(manifest: ScenarioManifest) -> tuple[int, tuple[int, int]]:
    base_frames = _read_video(manifest.base_video)
    events = []
    for spec in manifest.events:
        if isinstance(spec, InsertVideoSpec):
            inserted = _read_video(spec.path, spec.max_frames)
            events.append(InsertFrames(spec.at_index, tuple(inserted)))
        elif isinstance(spec, RepeatFrameSpec):
            events.append(RepeatFrame(spec.source_index, spec.at_index, spec.count))
    frames = apply_events(base_frames, events)
    if not frames:
        raise ValueError("Scenario produced no frames")
    dimensions = _write_video(manifest.output_video, frames, manifest.fps)
    return len(frames), dimensions


def _read_video(path: Path, limit: int | None = None) -> list[FramePacket]:
    source = video_source(str(path))
    frames: list[FramePacket] = []
    try:
        while limit is None or len(frames) < limit:
            frame = source.read()
            if frame is None:
                break
            frames.append(frame)
    finally:
        source.close()
    return frames


def _write_video(path: Path, frames: list[FramePacket], fps: float) -> tuple[int, int]:
    height, width = frames[0].image_bgr.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create scenario video: {path}")
    try:
        for frame in frames:
            if frame.image_bgr.shape[:2] != (height, width):
                raise ValueError(
                    "All scenario videos must have the same frame dimensions"
                )
            writer.write(frame.image_bgr)
    finally:
        writer.release()
    return width, height
