from __future__ import annotations

from src.face_auth.domain import reason_codes
from src.face_auth.domain.types import FramePacket, GateResult, GateStatus


class FrameIntegrityGate:
    def __init__(self) -> None:
        self._last_frame_id: int | None = None
        self._last_capture_time: float | None = None

    def evaluate(self, frame: FramePacket) -> GateResult:
        valid = True
        if self._last_frame_id is not None and frame.frame_id <= self._last_frame_id:
            valid = False
        if (
            self._last_capture_time is not None
            and frame.captured_at_monotonic <= self._last_capture_time
        ):
            valid = False

        if not valid:
            return GateResult(
                gate="frame_integrity",
                status=GateStatus.FAIL,
                reason_codes=(reason_codes.FRAME_SEQUENCE_INVALID,),
            )

        self._last_frame_id = frame.frame_id
        self._last_capture_time = frame.captured_at_monotonic
        return GateResult(gate="frame_integrity", status=GateStatus.PASS)

    def reset(self) -> None:
        self._last_frame_id = None
        self._last_capture_time = None
