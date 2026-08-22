import unittest

import numpy as np

from src.face_auth.adapters.in_memory_store import InMemoryStore
from src.face_auth.application.evidence_service import (
    build_capture_manifest,
    verify_capture_manifest,
)
from src.face_auth.application.session_service import SessionService
from src.face_auth.domain.types import FramePacket


def frame(frame_id, value=0):
    return FramePacket(
        frame_id,
        float(frame_id + 1),
        np.full((8, 8, 3), value, dtype=np.uint8),
    )


class EvidenceServiceTest(unittest.TestCase):
    def setUp(self):
        self.sessions = SessionService(InMemoryStore())
        self.session = self.sessions.create_session(user_id="user-1", purpose="TEST")
        self.sessions.issue_challenge(self.session.session_id)
        self.frames = [frame(0), frame(1, 1)]

    def test_manifest_is_bound_to_session_nonce_and_frame_bytes(self):
        manifest = build_capture_manifest(self.session, self.frames)
        self.assertTrue(verify_capture_manifest(manifest, self.session, self.frames))
        changed = [frame(0), frame(1, 2)]
        self.assertFalse(verify_capture_manifest(manifest, self.session, changed))

    def test_manifest_cannot_be_reused_with_another_session(self):
        manifest = build_capture_manifest(self.session, self.frames)
        other = self.sessions.create_session(user_id="user-1", purpose="TEST")
        self.sessions.issue_challenge(other.session_id)
        self.assertFalse(verify_capture_manifest(manifest, other, self.frames))


if __name__ == "__main__":
    unittest.main()
