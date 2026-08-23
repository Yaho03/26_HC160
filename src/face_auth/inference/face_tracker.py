from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackObservation:
    track_id: int | None
    switched: bool
    visible: bool


class SingleFaceTracker:
    """Lightweight bbox tracker used as a signal, not as identity proof."""

    def __init__(self, min_iou: float = 0.2) -> None:
        self.min_iou = min_iou
        self._bbox: tuple[int, int, int, int] | None = None
        self._track_id = 0

    def update(self, bbox: tuple[int, int, int, int] | None) -> TrackObservation:
        if bbox is None:
            self._bbox = None
            return TrackObservation(None, switched=False, visible=False)
        switched = self._bbox is not None and _iou(self._bbox, bbox) < self.min_iou
        if self._bbox is None or switched:
            self._track_id += 1
        self._bbox = bbox
        return TrackObservation(self._track_id, switched=switched, visible=True)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection_width = max(0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0
