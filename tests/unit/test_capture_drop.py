"""캡처 경로의 frame drop.

BACKLOG PERF-001 은 FPS, drop, P95 를 요구한다. P95 는 측정됐고 drop 이 남았다.

게이트 지연은 FPS 를 제한하지 않는다. FullEvidencePipeline.evaluate 가 inspector 를
인증 판정당 1회 호출하며 프레임 루프 안이 아니기 때문이다. 캡처 FPS 를 제한하는 것은
카메라 read 와 얼굴 검출이다. 두 값은 단위와 의미가 다르므로 섞지 않는다.

여기서는 카메라 없이 잴 수 있는 버퍼 동작만 다룬다.
"""

import unittest

from src.face_auth.adapters.capture_base import LatestFrameBuffer
from src.verification.defenses.capture_drop import (
    drop_profile,
    sustainable_fps,
)


class _Packet:
    def __init__(self, frame_id):
        self.frame_id = frame_id


class SustainableFpsTest(unittest.TestCase):
    """소비가 생산을 따라가지 못하면 드롭이 생긴다."""

    def test_fps_is_the_inverse_of_processing_time(self):
        self.assertAlmostEqual(sustainable_fps(415.0), 1000 / 415.0)

    def test_zero_processing_time_is_undefined(self):
        """분모가 0이면 0이 아니라 undefined다."""
        self.assertIsNone(sustainable_fps(0.0))

    def test_negative_processing_time_is_rejected(self):
        with self.assertRaises(ValueError):
            sustainable_fps(-1.0)


class DropProfileTest(unittest.TestCase):
    def test_no_drop_when_consumer_keeps_up(self):
        profile = drop_profile(produced=10, buffer_size=30, consume_every=1)
        self.assertEqual(profile["dropped"], 0)
        self.assertEqual(profile["delivered"], 10)

    def test_drop_starts_after_the_buffer_fills(self):
        """버퍼가 차기 전에는 드롭이 없다. 상한이 있다는 것이 이 버퍼의 설계다."""
        profile = drop_profile(produced=35, buffer_size=30, consume_every=0)
        self.assertEqual(profile["dropped"], 5)

    def test_drop_rate_is_reported_with_its_denominator(self):
        profile = drop_profile(produced=100, buffer_size=10, consume_every=0)
        self.assertEqual(profile["produced"], 100)
        self.assertAlmostEqual(profile["drop_rate"], profile["dropped"] / 100)

    def test_pop_latest_discards_the_backlog(self):
        """pop_latest 는 최신 한 장만 주고 나머지를 버린다. 실시간 인증의 설계다."""
        profile = drop_profile(produced=20, buffer_size=30, consume_every=10)
        self.assertLess(profile["delivered"], profile["produced"])

    def test_buffer_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            drop_profile(produced=10, buffer_size=0, consume_every=1)


class BufferBehaviourTest(unittest.TestCase):
    """LatestFrameBuffer 자체의 계약. 측정이 기대는 성질이다."""

    def test_counter_increases_only_after_the_buffer_is_full(self):
        buffer = LatestFrameBuffer(max_frames=3)
        for index in range(3):
            buffer.push(_Packet(index))
        self.assertEqual(buffer.dropped_frames, 0)

        buffer.push(_Packet(3))
        self.assertEqual(buffer.dropped_frames, 1)

    def test_pop_latest_returns_the_newest_and_clears(self):
        buffer = LatestFrameBuffer(max_frames=5)
        for index in range(4):
            buffer.push(_Packet(index))

        latest = buffer.pop_latest()
        self.assertEqual(latest.frame_id, 3)
        self.assertEqual(len(buffer), 0)

    def test_pop_on_empty_buffer_returns_none(self):
        self.assertIsNone(LatestFrameBuffer(max_frames=2).pop_latest())


if __name__ == "__main__":
    unittest.main()
