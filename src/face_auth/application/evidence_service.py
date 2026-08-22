from __future__ import annotations

import hashlib
import struct
import uuid

import numpy as np

from src.face_auth.domain.types import CaptureManifest, FaceAuthSession, FramePacket


def build_capture_manifest(
    session: FaceAuthSession,
    frames: list[FramePacket],
    *,
    dropped_frame_count: int = 0,
) -> CaptureManifest:
    if session.challenge is None:
        raise ValueError("Cannot bind evidence without a session challenge")
    if not frames:
        raise ValueError("Cannot build a manifest without frames")
    if dropped_frame_count < 0:
        raise ValueError("dropped_frame_count must be non-negative")
    return CaptureManifest(
        session_id=session.session_id,
        attempt_id=str(uuid.uuid4()),
        nonce=session.challenge.nonce,
        first_frame_id=frames[0].frame_id,
        last_frame_id=frames[-1].frame_id,
        captured_at_start_monotonic=frames[0].captured_at_monotonic,
        captured_at_end_monotonic=frames[-1].captured_at_monotonic,
        frame_count=len(frames),
        dropped_frame_count=dropped_frame_count,
        evidence_digest=evidence_digest(session.challenge.nonce, frames),
    )


def evidence_digest(nonce: str, frames: list[FramePacket]) -> str:
    digest = hashlib.sha256()
    digest.update(nonce.encode("utf-8"))
    for frame in frames:
        image = np.ascontiguousarray(frame.image_bgr)
        digest.update(struct.pack("!qd", frame.frame_id, frame.captured_at_monotonic))
        digest.update(str(image.shape).encode("ascii"))
        digest.update(str(image.dtype).encode("ascii"))
        digest.update(image.tobytes())
    return digest.hexdigest()


def verify_capture_manifest(
    manifest: CaptureManifest,
    session: FaceAuthSession,
    frames: list[FramePacket],
) -> bool:
    if session.challenge is None or not frames:
        return False
    return (
        manifest.session_id == session.session_id
        and manifest.nonce == session.challenge.nonce
        and manifest.first_frame_id == frames[0].frame_id
        and manifest.last_frame_id == frames[-1].frame_id
        and manifest.frame_count == len(frames)
        and manifest.evidence_digest == evidence_digest(session.challenge.nonce, frames)
    )
