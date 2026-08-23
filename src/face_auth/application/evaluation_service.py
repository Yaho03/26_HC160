from __future__ import annotations

from dataclasses import dataclass

from src.face_auth.adapters.in_memory_store import InMemoryStore
from src.face_auth.application.session_service import SessionService
from src.face_auth.application.token_service import TokenService
from src.face_auth.domain.policy import PolicyEngine
from src.face_auth.domain.types import (
    Decision,
    DecisionAction,
    GateResult,
    SessionState,
    VerificationTokenClaims,
)


@dataclass(frozen=True)
class EvaluationOutcome:
    decision: Decision
    token: VerificationTokenClaims | None = None


_DECISION_STATES = {
    DecisionAction.VERIFIED: SessionState.VERIFIED,
    DecisionAction.RETRYABLE: SessionState.RETRYABLE,
    DecisionAction.SECURITY_DENIED: SessionState.SECURITY_DENIED,
    DecisionAction.ERROR: SessionState.ERROR,
}


class EvaluationService:
    """Applies policy and commits exactly one terminal evaluation state."""

    def __init__(
        self,
        store: InMemoryStore,
        sessions: SessionService,
        policy: PolicyEngine,
        tokens: TokenService,
    ) -> None:
        self.store = store
        self.sessions = sessions
        self.policy = policy
        self.tokens = tokens

    def evaluate(
        self, session_id: str, gate_results: list[GateResult]
    ) -> EvaluationOutcome:
        session = self.store.get_session(session_id)
        if session.state is not SessionState.EVALUATING:
            raise ValueError(f"Session is not evaluating: {session.state.value}")

        decision = self.policy.evaluate(gate_results, session.security_profile)
        self.sessions.move(session_id, _DECISION_STATES[decision.action])
        token = (
            self.tokens.issue(session_id)
            if decision.action is DecisionAction.VERIFIED
            else None
        )
        return EvaluationOutcome(decision=decision, token=token)
