import unittest

from src.face_auth.domain import reason_codes
from src.face_auth.domain.policy import PolicyEngine
from src.face_auth.domain.types import (
    DecisionAction,
    GateResult,
    GateStatus,
    SecurityProfile,
)


def gate(name, status=GateStatus.PASS, *reasons):
    return GateResult(gate=name, status=status, reason_codes=tuple(reasons))


class PolicyEngineTest(unittest.TestCase):
    def setUp(self):
        self.policy = PolicyEngine("test-policy-v1")
        self.baseline_pass = [
            gate("frame_integrity"),
            gate("quality"),
            gate("single_face"),
            gate("identity"),
        ]

    def test_baseline_profile_can_verify_without_pad(self):
        decision = self.policy.evaluate(
            self.baseline_pass, SecurityProfile.BASELINE_ONLY
        )
        self.assertEqual(decision.action, DecisionAction.VERIFIED)

    def test_full_profile_requires_all_security_gates(self):
        decision = self.policy.evaluate(self.baseline_pass, SecurityProfile.FULL)
        self.assertEqual(decision.action, DecisionAction.ERROR)
        self.assertIn("MISSING_GATE:passive_pad", decision.reason_codes)

    def test_quality_failure_is_retryable(self):
        results = [
            gate("frame_integrity"),
            gate("quality", GateStatus.FAIL, reason_codes.BLUR),
            gate("single_face"),
            gate("identity"),
        ]
        decision = self.policy.evaluate(results, SecurityProfile.BASELINE_ONLY)
        self.assertEqual(decision.action, DecisionAction.RETRYABLE)

    def test_overexposure_is_retryable(self):
        results = [
            gate("frame_integrity"),
            gate("quality", GateStatus.FAIL, reason_codes.OVEREXPOSED),
            gate("single_face"),
            gate("identity"),
        ]
        decision = self.policy.evaluate(results, SecurityProfile.BASELINE_ONLY)
        self.assertEqual(decision.action, DecisionAction.RETRYABLE)

    def test_identity_failure_is_security_denied(self):
        results = [
            gate("frame_integrity"),
            gate("quality"),
            gate("single_face"),
            gate("identity", GateStatus.FAIL, reason_codes.LOW_IDENTITY_SIMILARITY),
        ]
        decision = self.policy.evaluate(results, SecurityProfile.BASELINE_ONLY)
        self.assertEqual(decision.action, DecisionAction.SECURITY_DENIED)

    def test_required_model_error_is_not_accepted(self):
        results = [
            gate("frame_integrity"),
            gate("quality"),
            gate("single_face"),
            gate("identity", GateStatus.ERROR, reason_codes.MODEL_ERROR),
        ]
        decision = self.policy.evaluate(results, SecurityProfile.BASELINE_ONLY)
        self.assertEqual(decision.action, DecisionAction.ERROR)

    def test_optional_adversarial_failure_vetoes_baseline(self):
        results = self.baseline_pass + [
            gate("adversarial", GateStatus.FAIL, reason_codes.ADVERSARIAL_SUSPECTED)
        ]
        decision = self.policy.evaluate(results, SecurityProfile.BASELINE_ONLY)
        self.assertEqual(decision.action, DecisionAction.SECURITY_DENIED)

    def test_upstream_multi_face_failure_beats_downstream_not_evaluated(self):
        results = [
            gate("frame_integrity"),
            gate("quality", GateStatus.NOT_EVALUATED),
            gate("single_face", GateStatus.FAIL, reason_codes.MULTIPLE_FACES),
            gate("identity", GateStatus.NOT_EVALUATED),
        ]
        decision = self.policy.evaluate(results, SecurityProfile.BASELINE_ONLY)
        self.assertEqual(decision.action, DecisionAction.SECURITY_DENIED)
        self.assertIn(reason_codes.MULTIPLE_FACES, decision.reason_codes)


if __name__ == "__main__":
    unittest.main()
