from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from src.face_auth.domain.types import FramePacket


@dataclass(frozen=True)
class InsertFrames:
    at_index: int
    frames: tuple[FramePacket, ...]


@dataclass(frozen=True)
class RepeatFrame:
    source_index: int
    at_index: int
    count: int


ScenarioEvent = Union[InsertFrames, RepeatFrame]


def apply_events(
    base_frames: list[FramePacket],
    events: list[ScenarioEvent],
) -> list[FramePacket]:
    """Build a deterministic sequence; inserted metadata is preserved for integrity tests."""
    output = list(base_frames)
    for event in events:
        if isinstance(event, InsertFrames):
            if not 0 <= event.at_index <= len(output):
                raise IndexError(f"Insert index out of range: {event.at_index}")
            output[event.at_index : event.at_index] = list(event.frames)
        elif isinstance(event, RepeatFrame):
            if not 0 <= event.source_index < len(output):
                raise IndexError(f"Source index out of range: {event.source_index}")
            if not 0 <= event.at_index <= len(output):
                raise IndexError(f"Repeat index out of range: {event.at_index}")
            output[event.at_index : event.at_index] = [
                output[event.source_index]
            ] * event.count
    return output
