from __future__ import annotations

from threading import RLock

from src.face_auth.domain.types import FaceAuthSession, VerificationTokenClaims


class InMemoryStore:
    """Thread-safe store used by tests and the first vertical slice."""

    def __init__(self) -> None:
        self._sessions: dict[str, FaceAuthSession] = {}
        self._tokens: dict[str, VerificationTokenClaims] = {}
        self._lock = RLock()

    def save_session(self, session: FaceAuthSession) -> None:
        with self._lock:
            self._sessions[session.session_id] = session

    def get_session(self, session_id: str) -> FaceAuthSession:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise KeyError(f"Unknown session: {session_id}") from exc

    def save_token(self, token: VerificationTokenClaims) -> None:
        with self._lock:
            self._tokens[token.token_id] = token

    def get_token(self, token_id: str) -> VerificationTokenClaims:
        with self._lock:
            try:
                return self._tokens[token_id]
            except KeyError as exc:
                raise KeyError(f"Unknown token: {token_id}") from exc

    def find_token_by_session(self, session_id: str) -> VerificationTokenClaims | None:
        with self._lock:
            return next(
                (
                    token
                    for token in self._tokens.values()
                    if token.session_id == session_id
                ),
                None,
            )

    @property
    def lock(self) -> RLock:
        return self._lock
