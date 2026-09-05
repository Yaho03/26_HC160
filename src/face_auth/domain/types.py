from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np


class SessionState(str, Enum):
    CREATED = "CREATED"
    CHALLENGE_ISSUED = "CHALLENGE_ISSUED"
    CAPTURING = "CAPTURING"
    EVIDENCE_RECEIVED = "EVIDENCE_RECEIVED"
    EVALUATING = "EVALUATING"
    VERIFIED = "VERIFIED"
    RETRYABLE = "RETRYABLE"
    SECURITY_DENIED = "SECURITY_DENIED"
    ERROR = "ERROR"
    CONSUMED = "CONSUMED"
    EXPIRED = "EXPIRED"


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUATED = "NOT_EVALUATED"
    ERROR = "ERROR"


class DecisionAction(str, Enum):
    VERIFIED = "VERIFIED"
    RETRYABLE = "RETRYABLE"
    SECURITY_DENIED = "SECURITY_DENIED"
    ERROR = "ERROR"


class SecurityProfile(str, Enum):
    BASELINE_ONLY = "BASELINE_ONLY"
    FULL = "FULL"


@dataclass(frozen=True)
class Challenge:
    nonce: str
    kind: str
    issued_at: datetime
    expires_at: datetime


@dataclass
class FaceAuthSession:
    session_id: str
    user_id: str
    purpose: str
    transaction_context_hash: str | None
    state: SessionState
    created_at: datetime
    expires_at: datetime
    policy_version: str
    security_profile: SecurityProfile
    attempt_count: int = 0
    challenge: Challenge | None = None


@dataclass(frozen=True)
class GateResult:
    gate: str
    status: GateStatus
    score: float | None = None
    threshold: float | None = None
    reason_codes: tuple[str, ...] = ()
    model_version: str | None = None
    threshold_version: str | None = None
    latency_ms: float = 0.0
    # Transformation별 evidence. 07_DEFENSE_AND_DETECTION_SPEC.md 6절이 요구한다.
    # 변환 파라미터가 무작위면 재현에 이 값이 필요하다. 고정 변환은 config에 이미
    # 적혀 있으므로 비워 둔다. 해시 가능해야 하므로 문자열 튜플을 쓴다.
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    action: DecisionAction
    gate_results: tuple[GateResult, ...]
    reason_codes: tuple[str, ...] = ()
    policy_version: str = ""


@dataclass
class VerificationTokenClaims:
    token_id: str
    session_id: str
    user_id: str
    purpose: str
    transaction_context_hash: str | None
    issued_at: datetime
    expires_at: datetime
    policy_version: str
    challenge_nonce: str
    consumed_at: datetime | None = None

    @property
    def consumed(self) -> bool:
        return self.consumed_at is not None


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    captured_at_monotonic: float
    image_bgr: np.ndarray = field(repr=False, compare=False)
    source_time_ms: float | None = None


@dataclass(frozen=True)
class CaptureManifest:
    session_id: str
    attempt_id: str
    nonce: str
    first_frame_id: int
    last_frame_id: int
    captured_at_start_monotonic: float
    captured_at_end_monotonic: float
    frame_count: int
    dropped_frame_count: int
    challenge_start_frame_id: int | None
    evidence_digest: str
