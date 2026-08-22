from __future__ import annotations

import secrets
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from src.face_auth.adapters.in_memory_store import InMemoryStore
from src.face_auth.domain.state_machine import transition
from src.face_auth.domain.types import (
    Challenge,
    FaceAuthSession,
    SecurityProfile,
    SessionState,
)


_CHALLENGES = ("HEAD_LEFT", "HEAD_RIGHT", "BLINK")


class SessionService:
    def __init__(
        self,
        store: InMemoryStore,
        *,
        session_ttl: timedelta = timedelta(minutes=3),
        challenge_ttl: timedelta = timedelta(seconds=30),
        policy_version: str = "face-auth-policy-v1",
        challenge_kinds: tuple[str, ...] = _CHALLENGES,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not challenge_kinds:
            raise ValueError("At least one challenge kind is required")
        self.store = store
        self.session_ttl = session_ttl
        self.challenge_ttl = challenge_ttl
        self.policy_version = policy_version
        self.challenge_kinds = challenge_kinds
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def create_session(
        self,
        *,
        user_id: str,
        purpose: str,
        transaction_context_hash: str | None = None,
        security_profile: SecurityProfile = SecurityProfile.FULL,
    ) -> FaceAuthSession:
        now = self.clock()
        session = FaceAuthSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            purpose=purpose,
            transaction_context_hash=transaction_context_hash,
            state=SessionState.CREATED,
            created_at=now,
            expires_at=now + self.session_ttl,
            policy_version=self.policy_version,
            security_profile=security_profile,
        )
        self.store.save_session(session)
        return session

    def issue_challenge(self, session_id: str) -> Challenge:
        session = self._active_session(session_id)
        session.state = transition(session.state, SessionState.CHALLENGE_ISSUED)
        now = self.clock()
        challenge = Challenge(
            nonce=secrets.token_urlsafe(24),
            kind=secrets.choice(self.challenge_kinds),
            issued_at=now,
            expires_at=min(now + self.challenge_ttl, session.expires_at),
        )
        session.challenge = challenge
        session.attempt_count += 1
        self.store.save_session(session)
        return challenge

    def move(self, session_id: str, target: SessionState) -> FaceAuthSession:
        session = self._active_session(session_id)
        if target is SessionState.CAPTURING:
            if session.challenge is None:
                raise ValueError("Cannot capture without a challenge")
            if self.clock() >= session.challenge.expires_at:
                session.state = transition(session.state, SessionState.EXPIRED)
                self.store.save_session(session)
                raise ValueError(f"Challenge expired: {session_id}")
        session.state = transition(session.state, target)
        self.store.save_session(session)
        return session

    def _active_session(self, session_id: str) -> FaceAuthSession:
        session = self.store.get_session(session_id)
        terminal = {
            SessionState.SECURITY_DENIED,
            SessionState.ERROR,
            SessionState.CONSUMED,
            SessionState.EXPIRED,
        }
        if self.clock() >= session.expires_at and session.state not in terminal:
            session.state = SessionState.EXPIRED
            self.store.save_session(session)
            raise ValueError(f"Session expired: {session_id}")
        return session
