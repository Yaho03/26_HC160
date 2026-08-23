from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.common.reproducibility import sha256_file, stable_json_bytes
from src.contracts.validation import ContractError, validate_relative_uri
from src.face_auth.domain.types import (
    DecisionAction,
    GateResult,
    SecurityProfile,
    SessionState,
)


SCHEMA_VERSION = "1.0"
_DECISION_STATES = {
    DecisionAction.VERIFIED: SessionState.VERIFIED,
    DecisionAction.RETRYABLE: SessionState.RETRYABLE,
    DecisionAction.SECURITY_DENIED: SessionState.SECURITY_DENIED,
    DecisionAction.ERROR: SessionState.ERROR,
}


class DecisionArtifactError(RuntimeError):
    """Raised when a decision artifact cannot be safely created."""


def build_decision_artifact(
    *,
    session_id: str,
    attempt_id: str,
    security_profile: SecurityProfile,
    state: SessionState,
    result_status: str,
    decision: DecisionAction,
    policy_version: str,
    reason_codes: Iterable[str],
    challenge_kind: str,
    challenge_start_frame_id: int | None,
    evidence_digest: str,
    total_frames: int,
    valid_face_frames: int | None,
    gate_results: Iterable[GateResult],
    token_id: str | None,
    decision_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    if not session_id or not attempt_id or not policy_version or not challenge_kind:
        raise DecisionArtifactError("decision identifiers must not be empty")
    if len(evidence_digest) != 64 or any(
        character not in "0123456789abcdef" for character in evidence_digest
    ):
        raise DecisionArtifactError("evidence_digest must be a lowercase SHA-256")
    if total_frames <= 0:
        raise DecisionArtifactError("total_frames must be positive")
    if challenge_start_frame_id is not None and challenge_start_frame_id < 0:
        raise DecisionArtifactError("challenge_start_frame_id must not be negative")
    if valid_face_frames is not None and not 0 <= valid_face_frames <= total_frames:
        raise DecisionArtifactError(
            "valid_face_frames must be between zero and total_frames"
        )
    if decision is not DecisionAction.VERIFIED and token_id is not None:
        raise DecisionArtifactError("only VERIFIED decisions may contain a token_id")
    if decision is DecisionAction.VERIFIED and not token_id:
        raise DecisionArtifactError("VERIFIED decisions require a token_id")
    if state is not _DECISION_STATES[decision]:
        raise DecisionArtifactError("terminal session state must match the decision")
    if result_status == "LIVE_SECURITY_VETO" and (
        decision is not DecisionAction.SECURITY_DENIED
        or security_profile is not SecurityProfile.FULL
    ):
        raise DecisionArtifactError(
            "LIVE_SECURITY_VETO requires a FULL SECURITY_DENIED decision"
        )
    if result_status not in {"POLICY_DECISION", "LIVE_SECURITY_VETO"}:
        raise DecisionArtifactError("unsupported decision result_status")

    return {
        "schema_version": SCHEMA_VERSION,
        "decision_id": decision_id or str(uuid.uuid4()),
        "created_at": created_at or _utc_now(),
        "session_id": session_id,
        "attempt_id": attempt_id,
        "security_profile": security_profile.value,
        "state": state.value,
        "result_status": result_status,
        "challenge": {
            "kind": challenge_kind,
            "start_frame_id": challenge_start_frame_id,
        },
        "evidence": {
            "sha256": evidence_digest,
            "frame_count": total_frames,
            "valid_face_frames": valid_face_frames,
        },
        "decision": {
            "action": decision.value,
            "policy_version": policy_version,
            "reason_codes": _unique_strings(reason_codes),
            "token_id": token_id,
        },
        "gates": [_gate_payload(result) for result in gate_results],
    }


def validate_decision_output(
    path: str | Path,
    *,
    overwrite: bool = False,
    protected_inputs: Iterable[str | Path] = (),
) -> Path:
    target = Path(path)
    if target.resolve() in {Path(value).resolve() for value in protected_inputs}:
        raise DecisionArtifactError("decision output must differ from input files")
    if target.exists() and not overwrite:
        raise DecisionArtifactError(
            f"refusing to overwrite existing decision artifact: {target}"
        )
    if target.exists() and not target.is_file():
        raise DecisionArtifactError(f"decision output is not a file: {target}")
    parent = target.parent
    if parent.exists() and not parent.is_dir():
        raise DecisionArtifactError(f"decision output parent is not a directory: {parent}")
    return target


def write_decision_artifact(
    path: str | Path,
    payload: dict[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    target = validate_decision_output(path, overwrite=overwrite)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = stable_json_bytes(payload) + b"\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, target)
            temporary_path = None
        else:
            try:
                os.link(temporary_path, target)
            except FileExistsError as error:
                raise DecisionArtifactError(
                    f"refusing to overwrite existing decision artifact: {target}"
                ) from error
        return target
    except OSError as error:
        raise DecisionArtifactError(
            f"unable to write decision artifact {target}: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_artifact_reference(
    path: str | Path,
    *,
    artifact_id: str,
    relative_uri: str,
    created_at: str,
    producer_run_id: str | None = None,
) -> dict[str, Any]:
    target = Path(path)
    if not artifact_id:
        raise DecisionArtifactError("artifact_id must not be empty")
    try:
        validate_relative_uri(relative_uri)
    except ContractError as error:
        raise DecisionArtifactError(str(error)) from error
    return {
        "schema_version": "1.0",
        "artifact_id": artifact_id,
        "kind": "decision",
        "relative_uri": relative_uri,
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
        "mime_type": "application/json",
        "sensitivity": "internal",
        "encryption": "none",
        "created_at": created_at,
        "parent_artifact_id": None,
        "retention_until": None,
        "producer_run_id": producer_run_id,
    }


def _gate_payload(result: GateResult) -> dict[str, Any]:
    return {
        "gate": result.gate,
        "status": result.status.value,
        "score": result.score,
        "threshold": result.threshold,
        "reason_codes": _unique_strings(result.reason_codes),
        "model_version": result.model_version,
        "threshold_version": result.threshold_version,
        "latency_ms": result.latency_ms,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _unique_strings(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
