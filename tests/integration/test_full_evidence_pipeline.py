import unittest

import cv2
import numpy as np
from PIL import Image

from src.face_auth.domain.policy import PolicyEngine
from src.face_auth.domain.types import DecisionAction, FramePacket, SecurityProfile
from src.face_auth.inference.active_liveness import ActiveLivenessGate
from src.face_auth.inference.camera_motion import CameraMotionConfig, CameraMotionGate
from src.face_auth.inference.continuity import ContinuityConfig, IdentityContinuityGate
from src.face_auth.inference.face_detector import DetectedFace
from src.face_auth.inference.full_pipeline import FullEvidencePipeline
from src.face_auth.inference.head_pose import FivePointHeadPoseEstimator
from src.face_auth.inference.passive_pad import PassivePADConfig, PassivePADGate
from src.face_auth.inference.pipeline import BaselineEvidencePipeline
from src.face_auth.inference.quality import QualityConfig, QualityGate
from src.face_auth.inference.verifier import MultiFrameVerifier, VerificationConfig


class FakeDetector:
    def detect(self, frame_bgr):
        crop = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        return [
            DetectedFace(
                (55, 35, 105, 85),
                1.0,
                crop,
                ((65, 50), (85, 50), (94, 62), (68, 75), (82, 75)),
            )
        ]


class FakeEmbedder:
    model_version = "fake-v1"

    def embed(self, images):
        return [np.array([1.0, 0.0], dtype=np.float32) for _ in images]


class FakePADScorer:
    def score(self, crops):
        return [0.95 for _ in crops]


class FullEvidencePipelineTest(unittest.TestCase):
    def test_all_real_pipeline_boundaries_can_verify(self):
        rng = np.random.default_rng(17)
        gray = rng.integers(30, 220, size=(120, 160), dtype=np.uint8)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        frames = []
        for index in range(5):
            varied = np.clip(image.astype(np.int16) + index, 0, 255).astype(np.uint8)
            frames.append(FramePacket(index, float(index + 1), varied))
        embedder = FakeEmbedder()
        baseline = BaselineEvidencePipeline(
            FakeDetector(),
            embedder,
            MultiFrameVerifier(
                np.array([1.0, 0.0]),
                VerificationConfig(0.8, "identity-test-v1", "fake-v1"),
            ),
            quality=QualityGate(QualityConfig(min_blur_variance=1.0)),
            min_valid_frames=5,
        )
        pipeline = FullEvidencePipeline(
            baseline,
            FakePADScorer(),
            PassivePADGate(PassivePADConfig(0.8, "pad-test-v1", "pad-threshold-v1")),
            FivePointHeadPoseEstimator(),
            ActiveLivenessGate(),
            IdentityContinuityGate(
                np.array([1.0, 0.0]),
                ContinuityConfig(window_size=5, failures_required=3),
            ),
            camera_motion_gate=CameraMotionGate(
                CameraMotionConfig(min_valid_pairs=2, min_tracked_points=8)
            ),
        )
        observation = pipeline.evaluate(
            frames,
            challenge_kind="HEAD_RIGHT",
            challenge_start_frame_id=-1,
        )
        decision = PolicyEngine().evaluate(
            observation.gate_results, SecurityProfile.FULL
        )
        self.assertEqual(decision.action, DecisionAction.VERIFIED)


if __name__ == "__main__":
    unittest.main()
