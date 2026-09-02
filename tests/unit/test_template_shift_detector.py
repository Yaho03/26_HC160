"""B안 두 번째 게이트: 등록 템플릿 기준 이동량 detector.

첫 게이트(transform-consistency)는 "입력에 조작 흔적이 있는가"를 보고, 이 게이트는
"그 조작이 등록자로 위장하는 방향인가"를 본다. 서로 다른 것을 재므로 독립 게이트로
유지한다. 근거는 docs/experiments/EXP-DET-001-camera-squeeze-probe.md 1절.
"""

import unittest

import numpy as np
from PIL import Image

from src.face_auth.domain.types import GateStatus
from src.face_auth.inference.adversarial_detector import (
    AdversarialDetectorConfig,
    TemplateShiftDetector,
    TemplateShiftDetectorConfig,
    TransformConsistencyDetector,
)
from src.face_auth.inference.feature_squeeze import FeatureSqueezeInspector


def _unit(*components):
    vector = np.asarray(components, dtype=np.float64)
    return vector / np.linalg.norm(vector)


class TemplateShiftDetectorTest(unittest.TestCase):
    def _detector(self, threshold, template):
        return TemplateShiftDetector(
            TemplateShiftDetectorConfig(
                max_template_shift=threshold, threshold_version="test-v1"
            ),
            template,
        )

    def test_gate_name_is_distinct_from_the_consistency_gate(self):
        detector = self._detector(0.1, _unit(1, 0))
        result = detector.evaluate(_unit(1, 0), [_unit(1, 0), _unit(1, 0)])

        self.assertEqual(result.gate, "adversarial_template")

    def test_score_is_the_largest_shift_towards_or_away_from_the_template(self):
        template = _unit(1, 0)
        original = _unit(1, 0)                       # cos(원본, 템플릿) = 1
        transformed = [_unit(0, 1), _unit(1, 0)]     # cos = 0, 1
        result = self._detector(10.0, template).evaluate(original, transformed)

        self.assertAlmostEqual(result.score, 1.0)

    def test_fails_when_shift_reaches_the_threshold(self):
        template = _unit(1, 0)
        result = self._detector(0.5, template).evaluate(
            _unit(1, 0), [_unit(0, 1), _unit(1, 0)]
        )

        self.assertIs(result.status, GateStatus.FAIL)
        self.assertIn("ADVERSARIAL_SUSPECTED", result.reason_codes)

    def test_passes_when_transforms_barely_move_the_template_similarity(self):
        template = _unit(1, 0)
        result = self._detector(0.5, template).evaluate(
            _unit(1, 0), [_unit(1, 0), _unit(1, 0)]
        )

        self.assertIs(result.status, GateStatus.PASS)

    def test_not_evaluated_without_enough_transforms(self):
        result = self._detector(0.5, _unit(1, 0)).evaluate(_unit(1, 0), [_unit(1, 0)])
        self.assertIs(result.status, GateStatus.NOT_EVALUATED)


class StubEmbedder:
    def __init__(self, vector):
        self.vector = vector
        self.calls = 0

    def embed(self, images):
        self.calls += 1
        return [self.vector for _ in images]


class DualGateInspectorTest(unittest.TestCase):
    def _crop(self):
        pixels = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        return Image.fromarray(pixels)

    def _inspector(self, embedder, template=None):
        consistency = TransformConsistencyDetector(
            AdversarialDetectorConfig(max_cosine_distance=0.5, threshold_version="c-v1")
        )
        template_detector = (
            TemplateShiftDetector(
                TemplateShiftDetectorConfig(
                    max_template_shift=0.5, threshold_version="t-v1"
                ),
                template,
            )
            if template is not None
            else None
        )
        return FeatureSqueezeInspector(
            embedder, consistency, template_detector=template_detector
        )

    def test_returns_only_the_consistency_gate_when_no_template_detector(self):
        inspector = self._inspector(StubEmbedder(_unit(1, 0)))
        results = inspector.evaluate([self._crop()], [_unit(1, 0)])

        self.assertEqual([result.gate for result in results], ["adversarial"])

    def test_returns_both_gates_when_a_template_detector_is_present(self):
        inspector = self._inspector(StubEmbedder(_unit(1, 0)), template=_unit(1, 0))
        results = inspector.evaluate([self._crop()], [_unit(1, 0)])

        self.assertEqual(
            [result.gate for result in results], ["adversarial", "adversarial_template"]
        )

    def test_transforms_and_embeddings_are_computed_once_for_both_gates(self):
        """두 게이트가 같은 변환·임베딩을 공유해야 한다. 두 배로 돌리면 안 된다."""
        embedder = StubEmbedder(_unit(1, 0))
        inspector = self._inspector(embedder, template=_unit(1, 0))
        inspector.evaluate([self._crop()], [_unit(1, 0)])

        self.assertEqual(embedder.calls, 1)


if __name__ == "__main__":
    unittest.main()


class PolicyIntegrationTest(unittest.TestCase):
    """두 게이트가 각각 독립적으로 거부권을 갖는지 확인한다.

    둘 다 optional 게이트이므로 필수 게이트가 모두 통과한 뒤에만 평가되고,
    둘 중 하나라도 FAIL이면 SECURITY_DENIED가 되어야 한다.
    """

    def _required_passes(self):
        from src.face_auth.domain.types import GateResult, GateStatus

        return [
            GateResult(name, GateStatus.PASS)
            for name in (
                "frame_integrity",
                "quality",
                "single_face",
                "identity",
                "camera_motion",
                "content_replay",
                "passive_pad",
                "active_liveness",
                "continuity",
            )
        ]

    def _decide(self, *optional):
        from src.face_auth.domain.policy import PolicyEngine
        from src.face_auth.domain.types import SecurityProfile

        return PolicyEngine().evaluate(
            self._required_passes() + list(optional), SecurityProfile.FULL
        )

    def test_both_gates_passing_verifies(self):
        from src.face_auth.domain.types import DecisionAction, GateResult, GateStatus

        decision = self._decide(
            GateResult("adversarial", GateStatus.PASS),
            GateResult("adversarial_template", GateStatus.PASS),
        )
        self.assertIs(decision.action, DecisionAction.VERIFIED)

    def test_template_gate_alone_can_deny(self):
        """첫 게이트가 통과해도 두 번째가 막을 수 있어야 B안이 의미를 갖는다."""
        from src.face_auth.domain import reason_codes
        from src.face_auth.domain.types import DecisionAction, GateResult, GateStatus

        decision = self._decide(
            GateResult("adversarial", GateStatus.PASS),
            GateResult(
                "adversarial_template",
                GateStatus.FAIL,
                reason_codes=(reason_codes.ADVERSARIAL_SUSPECTED,),
            ),
        )
        self.assertIs(decision.action, DecisionAction.SECURITY_DENIED)

    def test_consistency_gate_alone_can_deny(self):
        from src.face_auth.domain import reason_codes
        from src.face_auth.domain.types import DecisionAction, GateResult, GateStatus

        decision = self._decide(
            GateResult(
                "adversarial",
                GateStatus.FAIL,
                reason_codes=(reason_codes.ADVERSARIAL_SUSPECTED,),
            ),
            GateResult("adversarial_template", GateStatus.PASS),
        )
        self.assertIs(decision.action, DecisionAction.SECURITY_DENIED)

    def test_neither_gate_wired_still_verifies(self):
        """두 게이트 모두 optional이므로 배선하지 않아도 인증은 진행된다."""
        from src.face_auth.domain.types import DecisionAction

        self.assertIs(self._decide().action, DecisionAction.VERIFIED)
