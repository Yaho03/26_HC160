from __future__ import annotations

from .types import SessionState


class InvalidStateTransition(ValueError):
    pass


_ALLOWED_TRANSITIONS: dict[SessionState, frozenset[SessionState]] = {
    SessionState.CREATED: frozenset(
        {SessionState.CHALLENGE_ISSUED, SessionState.EXPIRED}
    ),
    SessionState.CHALLENGE_ISSUED: frozenset(
        {SessionState.CAPTURING, SessionState.EXPIRED, SessionState.ERROR}
    ),
    SessionState.CAPTURING: frozenset(
        {
            SessionState.EVIDENCE_RECEIVED,
            SessionState.RETRYABLE,
            SessionState.ERROR,
            SessionState.EXPIRED,
        }
    ),
    SessionState.EVIDENCE_RECEIVED: frozenset(
        {SessionState.EVALUATING, SessionState.ERROR, SessionState.EXPIRED}
    ),
    SessionState.EVALUATING: frozenset(
        {
            SessionState.VERIFIED,
            SessionState.RETRYABLE,
            SessionState.SECURITY_DENIED,
            SessionState.ERROR,
            SessionState.EXPIRED,
        }
    ),
    SessionState.RETRYABLE: frozenset(
        {
            SessionState.CHALLENGE_ISSUED,
            SessionState.SECURITY_DENIED,
            SessionState.EXPIRED,
        }
    ),
    SessionState.VERIFIED: frozenset({SessionState.CONSUMED, SessionState.EXPIRED}),
    SessionState.SECURITY_DENIED: frozenset(),
    SessionState.ERROR: frozenset(),
    SessionState.CONSUMED: frozenset(),
    SessionState.EXPIRED: frozenset(),
}


def can_transition(current: SessionState, target: SessionState) -> bool:
    return target in _ALLOWED_TRANSITIONS[current]


def transition(current: SessionState, target: SessionState) -> SessionState:
    if not can_transition(current, target):
        raise InvalidStateTransition(
            f"Cannot transition from {current.value} to {target.value}"
        )
    return target
