from __future__ import annotations

import secrets
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from src.face_auth.adapters.in_memory_store import InMemoryStore
from src.face_auth.domain import reason_codes
from src.face_auth.domain.state_machine import transition
from src.face_auth.domain.types import SessionState, VerificationTokenClaims


class TokenValidationError(ValueError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


class TokenService:
    def __init__(
        self,
        store: InMemoryStore,
        *,
        token_ttl: timedelta = timedelta(seconds=90),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.token_ttl = token_ttl
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def issue(self, session_id: str) -> VerificationTokenClaims:
        session = self.store.get_session(session_id)
        if session.state is not SessionState.VERIFIED:
            raise TokenValidationError("SESSION_NOT_VERIFIED")
        existing = self.store.find_token_by_session(session_id)
        if existing is not None:
            return existing
        now = self.clock()
        if session.challenge is None:
            raise TokenValidationError("CHALLENGE_NOT_ISSUED")
        token = VerificationTokenClaims(
            token_id=secrets.token_urlsafe(24),
            session_id=session.session_id,
            user_id=session.user_id,
            purpose=session.purpose,
            transaction_context_hash=session.transaction_context_hash,
            issued_at=now,
            expires_at=min(now + self.token_ttl, session.expires_at),
            policy_version=session.policy_version,
            challenge_nonce=session.challenge.nonce,
        )
        self.store.save_token(token)
        return token

    def consume(
        self,
        token_id: str,
        *,
        user_id: str,
        purpose: str,
        transaction_context_hash: str | None,
    ) -> VerificationTokenClaims:
        with self.store.lock:
            token = self.store.get_token(token_id)
            now = self.clock()
            if token.consumed:
                raise TokenValidationError(reason_codes.TOKEN_ALREADY_CONSUMED)
            if now >= token.expires_at:
                raise TokenValidationError(reason_codes.SESSION_EXPIRED)
            if (
                token.user_id != user_id
                or token.purpose != purpose
                or token.transaction_context_hash != transaction_context_hash
            ):
                raise TokenValidationError(reason_codes.CONTEXT_MISMATCH)

            session = self.store.get_session(token.session_id)
            session.state = transition(session.state, SessionState.CONSUMED)
            token.consumed_at = now
            self.store.save_session(session)
            self.store.save_token(token)
            return token
