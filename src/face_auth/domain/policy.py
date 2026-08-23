from __future__ import annotations

from collections.abc import Iterable

from . import reason_codes
from .types import Decision, DecisionAction, GateResult, GateStatus, SecurityProfile


_BASELINE_REQUIRED = frozenset(
    {"frame_integrity", "quality", "single_face", "identity"}
)
_FULL_REQUIRED = _BASELINE_REQUIRED | frozenset(
    {"camera_motion", "content_replay", "passive_pad", "active_liveness", "continuity"}
)
_RETRYABLE_REASONS = frozenset(
    {
        reason_codes.NO_FACE,
        reason_codes.BLUR,
        reason_codes.LOW_LIGHT,
        reason_codes.OVEREXPOSED,
        reason_codes.CAMERA_SHAKE,
        reason_codes.EXTREME_POSE,
        reason_codes.INSUFFICIENT_VALID_FRAMES,
    }
)


class PolicyEngine:
    def __init__(self, policy_version: str = "face-auth-policy-v1") -> None:
        self.policy_version = policy_version

    def evaluate(
        self,
        results: Iterable[GateResult],
        profile: SecurityProfile,
    ) -> Decision:
        ordered = tuple(results)
        by_gate = {result.gate: result for result in ordered}
        required = (
            _FULL_REQUIRED if profile is SecurityProfile.FULL else _BASELINE_REQUIRED
        )

        missing = sorted(required - by_gate.keys())
        if missing:
            return Decision(
                action=DecisionAction.ERROR,
                gate_results=ordered,
                reason_codes=tuple(f"MISSING_GATE:{gate}" for gate in missing),
                policy_version=self.policy_version,
            )

        required_results = tuple(by_gate[gate] for gate in sorted(required))
        errored = [
            result for result in required_results if result.status is GateStatus.ERROR
        ]
        if errored:
            reasons = self._collect_reasons(errored)
            if not reasons:
                reasons = tuple(
                    f"{result.gate}:{result.status.value}" for result in errored
                )
            return Decision(
                action=DecisionAction.ERROR,
                gate_results=ordered,
                reason_codes=reasons,
                policy_version=self.policy_version,
            )

        failed = [
            result for result in required_results if result.status is GateStatus.FAIL
        ]
        if failed:
            reasons = self._collect_reasons(failed)
            action = (
                DecisionAction.RETRYABLE
                if reasons and set(reasons).issubset(_RETRYABLE_REASONS)
                else DecisionAction.SECURITY_DENIED
            )
            return Decision(
                action=action,
                gate_results=ordered,
                reason_codes=reasons,
                policy_version=self.policy_version,
            )

        not_evaluated = [
            result
            for result in required_results
            if result.status is GateStatus.NOT_EVALUATED
        ]
        if not_evaluated:
            reasons = self._collect_reasons(not_evaluated)
            if not reasons:
                reasons = tuple(
                    f"{result.gate}:{result.status.value}" for result in not_evaluated
                )
            return Decision(
                action=DecisionAction.ERROR,
                gate_results=ordered,
                reason_codes=reasons,
                policy_version=self.policy_version,
            )

        # Optional security gates can still veto a baseline decision when evaluated.
        optional_failures = [
            result
            for result in ordered
            if result.gate not in required and result.status is GateStatus.FAIL
        ]
        if optional_failures:
            return Decision(
                action=DecisionAction.SECURITY_DENIED,
                gate_results=ordered,
                reason_codes=self._collect_reasons(optional_failures),
                policy_version=self.policy_version,
            )

        return Decision(
            action=DecisionAction.VERIFIED,
            gate_results=ordered,
            policy_version=self.policy_version,
        )

    @staticmethod
    def _collect_reasons(results: Iterable[GateResult]) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(code for result in results for code in result.reason_codes)
        )
