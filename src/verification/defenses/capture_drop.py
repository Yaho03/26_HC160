"""
캡처 경로의 frame drop 측정

BACKLOG PERF-001은 FPS, drop, P95를 요구한다. P95는 latency_bench가 측정했고 drop이
남았다.

게이트 지연과 캡처 FPS를 섞지 않는다. FullEvidencePipeline.evaluate는 inspector를
인증 판정당 1회 호출하며 프레임 루프 안이 아니다. 따라서 게이트는 캡처 FPS를 제한하지
않고 인증 1회당 종단 지연만 더한다. 캡처 FPS를 제한하는 것은 카메라 read와 얼굴
검출이다.

이 모듈은 카메라 없이 잴 수 있는 버퍼 동작만 다룬다. 실제 드라이버 드롭과 카메라
실효 FPS는 기기가 있어야 하며 미측정으로 남긴다.
"""

from __future__ import annotations

from src.face_auth.adapters.capture_base import LatestFrameBuffer


class _Packet:
    """버퍼가 프레임 내용을 보지 않으므로 식별자만 담는다."""

    __slots__ = ("frame_id",)

    def __init__(self, frame_id: int) -> None:
        self.frame_id = frame_id


def sustainable_fps(processing_ms: float) -> float | None:
    """
    한 프레임 처리에 걸리는 시간에서 지속 가능한 FPS 상한을 낸다.

    분모가 0이면 0이 아니라 undefined를 돌려준다. 09_EVALUATION_METRICS.md 1절의
    분모 규칙과 같다.
    """
    if processing_ms < 0:
        raise ValueError("처리 시간은 음수일 수 없다")
    if processing_ms == 0:
        return None
    return 1000.0 / processing_ms


def drop_profile(*, produced: int, buffer_size: int, consume_every: int) -> dict:
    """
    생산과 소비 비율에 따른 드롭을 센다.

    consume_every가 0이면 소비자가 전혀 읽지 않는다. n이면 n프레임마다 한 번
    pop_latest한다. pop_latest는 최신 한 장만 주고 나머지를 버리므로, 소비가 느리면
    버퍼가 차기 전에도 프레임이 전달되지 않는다. 실시간 인증에서 오래된 프레임보다
    최신 프레임이 중요하다는 설계다.

    dropped는 버퍼 오버플로우로 밀려난 수이고, delivered는 소비자에게 실제로 전달된
    수다. 둘은 다른 것을 세므로 합이 produced가 되지 않는다.
    """
    if buffer_size < 1:
        raise ValueError("buffer_size는 1 이상이어야 한다")
    if produced < 0 or consume_every < 0:
        raise ValueError("produced와 consume_every는 음수일 수 없다")

    buffer = LatestFrameBuffer(max_frames=buffer_size)
    delivered = 0

    for index in range(produced):
        buffer.push(_Packet(index))
        if consume_every and (index + 1) % consume_every == 0:
            if buffer.pop_latest() is not None:
                delivered += 1

    return {
        "produced": produced,
        "buffer_size": buffer_size,
        "consume_every": consume_every,
        "dropped": buffer.dropped_frames,
        "delivered": delivered,
        "drop_rate": buffer.dropped_frames / produced if produced else None,
        "still_buffered": len(buffer),
    }
