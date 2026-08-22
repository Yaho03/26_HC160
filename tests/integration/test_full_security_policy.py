import unittest

from src.face_auth.domain import reason_codes
from src.face_auth.domain.policy import PolicyEngine
from src.face_auth.domain.types import (
    DecisionAction,
    GateResult,
    GateStatus,
    SecurityProfile,
)


class FullSecurityPolicyTest(unittest.TestCase):
    def full_pass(self):
        return [
            GateResult("frame_integrity", GateStatus.PASS),
            GateResult("quality", GateStatus.PASS),
            GateResult("single_face", GateStatus.PASS),
            GateResult("identity", GateStatus.PASS),
            GateResult("camera_motion", GateStatus.PASS),
            GateResult("content_replay", GateStatus.PASS),
            GateResult("passive_pad", GateStatus.PASS),
            GateResult("active_liveness", GateStatus.PASS),
            GateResult("continuity", GateStatus.PASS),
            GateResult("adversarial", GateStatus.PASS),
        ]

    def test_all_full_profile_gates_pass(self):
        decision = PolicyEngine().evaluate(self.full_pass(), SecurityProfile.FULL)
        self.assertEqual(decision.action, DecisionAction.VERIFIED)

    def test_pad_failure_denies_full_profile(self):
        results = self.full_pass()
        results[6] = GateResult(
            "passive_pad",
            GateStatus.FAIL,
            reason_codes=(reason_codes.SPOOF_SUSPECTED,),
        )
        decision = PolicyEngine().evaluate(results, SecurityProfile.FULL)
        self.assertEqual(decision.action, DecisionAction.SECURITY_DENIED)

    def test_adversarial_secondary_failure_vetoes_full_profile(self):
        results = self.full_pass()
        results[-1] = GateResult(
            "adversarial",
            GateStatus.FAIL,
            reason_codes=(reason_codes.ADVERSARIAL_SUSPECTED,),
        )
        decision = PolicyEngine().evaluate(results, SecurityProfile.FULL)
        self.assertEqual(decision.action, DecisionAction.SECURITY_DENIED)


if __name__ == "__main__":
    unittest.main()
