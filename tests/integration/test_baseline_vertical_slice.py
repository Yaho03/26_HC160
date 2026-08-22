import unittest

from src.face_auth.adapters.in_memory_store import InMemoryStore
from src.face_auth.application.evaluation_service import EvaluationService
from src.face_auth.application.session_service import SessionService
from src.face_auth.application.token_service import TokenService
from src.face_auth.domain import reason_codes
from src.face_auth.domain.policy import PolicyEngine
from src.face_auth.domain.types import (
    DecisionAction,
    GateResult,
    GateStatus,
    SecurityProfile,
    SessionState,
)


def passing_baseline():
    return [
        GateResult("frame_integrity", GateStatus.PASS),
        GateResult("quality", GateStatus.PASS),
        GateResult("single_face", GateStatus.PASS),
        GateResult("identity", GateStatus.PASS),
    ]


class BaselineVerticalSliceTest(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore()
        self.sessions = SessionService(self.store)
        self.tokens = TokenService(self.store)
        self.evaluation = EvaluationService(
            self.store,
            self.sessions,
            PolicyEngine("test-policy-v1"),
            self.tokens,
        )

    def start_evaluation(self):
        session = self.sessions.create_session(
            user_id="user-1",
            purpose="HIGH_RISK_ACTION",
            transaction_context_hash="context-a",
            security_profile=SecurityProfile.BASELINE_ONLY,
        )
        self.sessions.issue_challenge(session.session_id)
        for state in (
            SessionState.CAPTURING,
            SessionState.EVIDENCE_RECEIVED,
            SessionState.EVALUATING,
        ):
            self.sessions.move(session.session_id, state)
        return session

    def test_verified_session_issues_context_bound_token(self):
        session = self.start_evaluation()
        outcome = self.evaluation.evaluate(session.session_id, passing_baseline())
        self.assertEqual(outcome.decision.action, DecisionAction.VERIFIED)
        self.assertIsNotNone(outcome.token)
        self.tokens.consume(
            outcome.token.token_id,
            user_id="user-1",
            purpose="HIGH_RISK_ACTION",
            transaction_context_hash="context-a",
        )
        self.assertEqual(session.state, SessionState.CONSUMED)

    def test_blur_is_retryable_and_does_not_issue_token(self):
        session = self.start_evaluation()
        results = passing_baseline()
        results[1] = GateResult(
            "quality", GateStatus.FAIL, reason_codes=(reason_codes.BLUR,)
        )
        outcome = self.evaluation.evaluate(session.session_id, results)
        self.assertEqual(outcome.decision.action, DecisionAction.RETRYABLE)
        self.assertIsNone(outcome.token)
        self.assertEqual(session.state, SessionState.RETRYABLE)

    def test_identity_failure_is_denied_and_does_not_issue_token(self):
        session = self.start_evaluation()
        results = passing_baseline()
        results[3] = GateResult(
            "identity",
            GateStatus.FAIL,
            reason_codes=(reason_codes.LOW_IDENTITY_SIMILARITY,),
        )
        outcome = self.evaluation.evaluate(session.session_id, results)
        self.assertEqual(outcome.decision.action, DecisionAction.SECURITY_DENIED)
        self.assertIsNone(outcome.token)
        self.assertEqual(session.state, SessionState.SECURITY_DENIED)

    def test_model_error_never_issues_token(self):
        session = self.start_evaluation()
        results = passing_baseline()
        results[3] = GateResult(
            "identity", GateStatus.ERROR, reason_codes=(reason_codes.MODEL_ERROR,)
        )
        outcome = self.evaluation.evaluate(session.session_id, results)
        self.assertEqual(outcome.decision.action, DecisionAction.ERROR)
        self.assertIsNone(outcome.token)


if __name__ == "__main__":
    unittest.main()
