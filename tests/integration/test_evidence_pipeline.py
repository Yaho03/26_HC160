import unittest

import numpy as np
from PIL import Image

from src.face_auth.domain import reason_codes
from src.face_auth.domain.policy import PolicyEngine
from src.face_auth.domain.types import (
    DecisionAction,
    FramePacket,
    GateStatus,
    SecurityProfile,
)
from src.face_auth.inference.face_detector import DetectedFace
from src.face_auth.inference.pipeline import BaselineEvidencePipeline
from src.face_auth.inference.quality import QualityConfig, QualityGate
from src.face_auth.inference.verifier import MultiFrameVerifier, VerificationConfig


def textured_frame(marker=1, frame_id=0):
    checker = (np.indices((64, 64)).sum(axis=0) % 2 * 100 + 80).astype(np.uint8)
    image = np.repeat(checker[..., None], 3, axis=2)
    image[0, 0, 0] = marker
    return FramePacket(frame_id, float(frame_id + 1), image)


class FakeDetector:
    def detect(self, frame_bgr):
        marker = int(frame_bgr[0, 0, 0])
        crop = Image.fromarray(frame_bgr[:, :, ::-1])
        if marker == 0:
            return []
        count = 2 if marker == 2 else 1
        return [DetectedFace((0, 0, 64, 64), 1.0, crop) for _ in range(count)]


class FakeEmbedder:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)

    def embed(self, images):
        return [self.vector.copy() for _ in images]


def make_pipeline(vector=(1.0, 0.0)):
    verifier = MultiFrameVerifier(
        np.array([1.0, 0.0], dtype=np.float32),
        VerificationConfig(0.8, "test-threshold-v1", "fake-model-v1"),
    )
    return BaselineEvidencePipeline(
        FakeDetector(),
        FakeEmbedder(vector),
        verifier,
        quality=QualityGate(QualityConfig(min_blur_variance=1.0)),
    )


class EvidencePipelineTest(unittest.TestCase):
    def test_genuine_window_passes_baseline_policy(self):
        observation = make_pipeline().evaluate(
            [textured_frame(frame_id=i) for i in range(3)]
        )
        self.assertEqual(observation.valid_face_frames, 3)
        decision = PolicyEngine().evaluate(
            observation.gate_results, SecurityProfile.BASELINE_ONLY
        )
        self.assertEqual(decision.action, DecisionAction.VERIFIED)

    def test_impostor_window_is_denied(self):
        observation = make_pipeline((0.0, 1.0)).evaluate(
            [textured_frame(frame_id=i) for i in range(3)]
        )
        decision = PolicyEngine().evaluate(
            observation.gate_results, SecurityProfile.BASELINE_ONLY
        )
        self.assertEqual(decision.action, DecisionAction.SECURITY_DENIED)

    def test_multiple_faces_are_denied_not_reported_as_model_error(self):
        frames = [textured_frame(frame_id=0), textured_frame(marker=2, frame_id=1)]
        observation = make_pipeline().evaluate(frames)
        by_gate = {result.gate: result for result in observation.gate_results}
        self.assertEqual(by_gate["single_face"].status, GateStatus.FAIL)
        self.assertIn(reason_codes.MULTIPLE_FACES, by_gate["single_face"].reason_codes)
        decision = PolicyEngine().evaluate(
            observation.gate_results, SecurityProfile.BASELINE_ONLY
        )
        self.assertEqual(decision.action, DecisionAction.SECURITY_DENIED)

    def test_replayed_frame_metadata_fails_integrity(self):
        frames = [textured_frame(frame_id=i) for i in range(3)]
        frames[2] = FramePacket(1, 4.0, frames[2].image_bgr)
        observation = make_pipeline().evaluate(frames)
        by_gate = {result.gate: result for result in observation.gate_results}
        self.assertEqual(by_gate["frame_integrity"].status, GateStatus.FAIL)
        self.assertIn(
            reason_codes.FRAME_SEQUENCE_INVALID,
            by_gate["frame_integrity"].reason_codes,
        )


if __name__ == "__main__":
    unittest.main()
