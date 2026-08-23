import unittest
from datetime import datetime, timedelta, timezone

from src.face_auth.adapters.in_memory_store import InMemoryStore
from src.face_auth.application.session_service import SessionService
from src.face_auth.application.token_service import TokenService, TokenValidationError
from src.face_auth.domain import reason_codes
from src.face_auth.domain.types import SecurityProfile, SessionState


class MutableClock:
    def __init__(self):
        self.now = datetime(2026, 8, 22, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, delta):
        self.now += delta


class SessionAndTokenTest(unittest.TestCase):
    def setUp(self):
        self.clock = MutableClock()
        self.store = InMemoryStore()
        self.sessions = SessionService(self.store, clock=self.clock)
        self.tokens = TokenService(
            self.store, clock=self.clock, token_ttl=timedelta(seconds=30)
        )

    def verified_session(self):
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
            SessionState.VERIFIED,
        ):
            self.sessions.move(session.session_id, state)
        return session

    def test_challenge_is_issued_after_session_creation(self):
        session = self.sessions.create_session(
            user_id="user-1", purpose="ENROLL_DEVICE"
        )
        challenge = self.sessions.issue_challenge(session.session_id)
        self.assertTrue(challenge.nonce)
        self.assertIn(challenge.kind, {"HEAD_LEFT", "HEAD_RIGHT", "BLINK"})
        self.assertEqual(session.state, SessionState.CHALLENGE_ISSUED)

    def test_token_is_bound_and_consumed_once(self):
        session = self.verified_session()
        token = self.tokens.issue(session.session_id)
        self.assertEqual(token.challenge_nonce, session.challenge.nonce)
        consumed = self.tokens.consume(
            token.token_id,
            user_id="user-1",
            purpose="HIGH_RISK_ACTION",
            transaction_context_hash="context-a",
        )
        self.assertTrue(consumed.consumed)
        self.assertEqual(session.state, SessionState.CONSUMED)
        with self.assertRaises(TokenValidationError) as ctx:
            self.tokens.consume(
                token.token_id,
                user_id="user-1",
                purpose="HIGH_RISK_ACTION",
                transaction_context_hash="context-a",
            )
        self.assertEqual(ctx.exception.reason_code, reason_codes.TOKEN_ALREADY_CONSUMED)

    def test_changed_context_is_rejected_without_consuming(self):
        session = self.verified_session()
        token = self.tokens.issue(session.session_id)
        with self.assertRaises(TokenValidationError) as ctx:
            self.tokens.consume(
                token.token_id,
                user_id="user-1",
                purpose="HIGH_RISK_ACTION",
                transaction_context_hash="context-b",
            )
        self.assertEqual(ctx.exception.reason_code, reason_codes.CONTEXT_MISMATCH)
        self.assertFalse(token.consumed)
        self.assertEqual(session.state, SessionState.VERIFIED)

    def test_expired_token_is_rejected(self):
        session = self.verified_session()
        token = self.tokens.issue(session.session_id)
        self.clock.advance(timedelta(seconds=31))
        with self.assertRaises(TokenValidationError) as ctx:
            self.tokens.consume(
                token.token_id,
                user_id="user-1",
                purpose="HIGH_RISK_ACTION",
                transaction_context_hash="context-a",
            )
        self.assertEqual(ctx.exception.reason_code, reason_codes.SESSION_EXPIRED)

    def test_repeated_issue_is_idempotent_for_a_verified_session(self):
        session = self.verified_session()
        first = self.tokens.issue(session.session_id)
        second = self.tokens.issue(session.session_id)
        self.assertEqual(first.token_id, second.token_id)


if __name__ == "__main__":
    unittest.main()
