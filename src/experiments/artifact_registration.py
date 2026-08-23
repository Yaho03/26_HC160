"""Register a completed JSON output without inventing experiment metadata."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.common.reproducibility import sha256_file, stable_json_bytes
from src.contracts.validation import (
    ContractError,
    validate_relative_uri,
    validate_sha256,
)
from src.experiments.run_manifest import RunManifest


_EXPERIMENT_ID = re.compile(r"^EXP-[A-Z]+-[0-9]{3}$")
_REQUIREMENT_ID = re.compile(r"^[A-Z]+-[0-9]{3}$")
_GIT_COMMIT = re.compile(r"^[a-f0-9]{40}$")


class ArtifactRegistrationError(RuntimeError):
    """Raised when an output cannot be registered without data loss."""


@dataclass(frozen=True)
class RegistrationContext:
    run_id: str
    experiment_id: str
    requirement_ids: tuple[str, ...]
    environment_sha256: str
    seed: int
    input_artifact_ids: tuple[str, ...]
    reproduce_command: str
    artifact_id: str
    relative_uri: str
    notes: str | None = None
    schema_version: str = "1.0"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegistrationContext":
        allowed = {
            "schema_version",
            "run_id",
            "experiment_id",
            "requirement_ids",
            "environment_sha256",
            "seed",
            "input_artifact_ids",
            "reproduce_command",
            "artifact_id",
            "relative_uri",
            "notes",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ArtifactRegistrationError(
                f"unknown registration context fields: {', '.join(unknown)}"
            )
        required = allowed - {"notes"}
        missing = sorted(required - set(value))
        if missing:
            raise ArtifactRegistrationError(
                f"missing registration context fields: {', '.join(missing)}"
            )
        try:
            context = cls(
                schema_version=_string(value["schema_version"], "schema_version"),
                run_id=_string(value["run_id"], "run_id"),
                experiment_id=_string(value["experiment_id"], "experiment_id"),
                requirement_ids=_string_tuple(
                    value["requirement_ids"], "requirement_ids"
                ),
                environment_sha256=_string(
                    value["environment_sha256"], "environment_sha256"
                ),
                seed=_integer(value["seed"], "seed"),
                input_artifact_ids=_string_tuple(
                    value["input_artifact_ids"], "input_artifact_ids", allow_empty=True
                ),
                reproduce_command=_string(
                    value["reproduce_command"], "reproduce_command"
                ),
                artifact_id=_string(value["artifact_id"], "artifact_id"),
                relative_uri=_string(value["relative_uri"], "relative_uri"),
                notes=(
                    None
                    if value.get("notes") is None
                    else _string(value["notes"], "notes")
                ),
            )
        except (KeyError, TypeError) as error:
            raise ArtifactRegistrationError(
                "registration context must be a JSON object with valid field types"
            ) from error
        context.validate()
        return context

    def validate(self) -> None:
        if self.schema_version != "1.0":
            raise ArtifactRegistrationError("unsupported registration schema_version")
        if not _EXPERIMENT_ID.fullmatch(self.experiment_id):
            raise ArtifactRegistrationError("invalid experiment_id")
        if not self.requirement_ids or any(
            not _REQUIREMENT_ID.fullmatch(value) for value in self.requirement_ids
        ):
            raise ArtifactRegistrationError("invalid requirement_ids")
        if len(set(self.requirement_ids)) != len(self.requirement_ids):
            raise ArtifactRegistrationError("requirement_ids must be unique")
        if len(set(self.input_artifact_ids)) != len(self.input_artifact_ids):
            raise ArtifactRegistrationError("input_artifact_ids must be unique")
        if self.artifact_id in self.input_artifact_ids:
            raise ArtifactRegistrationError(
                "output artifact_id must differ from input artifact IDs"
            )
        if self.seed < 0:
            raise ArtifactRegistrationError("seed must be non-negative")
        try:
            validate_sha256(self.environment_sha256)
            validate_relative_uri(self.relative_uri)
        except ContractError as error:
            raise ArtifactRegistrationError(str(error)) from error


@dataclass(frozen=True)
class RegistrationOutputs:
    artifact_reference: Path
    run_manifest: Path


def load_registration_context(path: str | Path) -> RegistrationContext:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactRegistrationError(
            f"unable to load registration context {source}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ArtifactRegistrationError("registration context must be a JSON object")
    return RegistrationContext.from_dict(value)


def registration_outputs(artifact_path: str | Path) -> RegistrationOutputs:
    source = Path(artifact_path)
    return RegistrationOutputs(
        artifact_reference=source.with_name(
            f"{source.name}.artifact-reference.json"
        ),
        run_manifest=source.with_name(f"{source.name}.run-manifest.json"),
    )


def preflight_registration(artifact_path: str | Path) -> RegistrationOutputs:
    outputs = registration_outputs(artifact_path)
    for path in (outputs.artifact_reference, outputs.run_manifest):
        if path.exists():
            raise ArtifactRegistrationError(
                f"refusing to overwrite existing registration output: {path}"
            )
        if path.parent.exists() and not path.parent.is_dir():
            raise ArtifactRegistrationError(
                f"registration output parent is not a directory: {path.parent}"
            )
    return outputs


def register_completed_output(
    artifact_path: str | Path,
    *,
    context: RegistrationContext,
    kind: str,
    created_at: str,
    config_sha256: str,
    git_commit: str,
    dirty_worktree: bool,
    device: Mapping[str, Any],
    started_at: str,
    ended_at: str,
    mime_type: str = "application/json",
    sensitivity: str = "internal",
    encryption: str = "none",
) -> RegistrationOutputs:
    source = Path(artifact_path)
    if not source.is_file():
        raise ArtifactRegistrationError(
            f"artifact must exist before registration: {source}"
        )
    context.validate()
    try:
        validate_sha256(config_sha256)
    except ContractError as error:
        raise ArtifactRegistrationError(str(error)) from error
    if not _GIT_COMMIT.fullmatch(git_commit):
        raise ArtifactRegistrationError(
            "git_commit must be 40 lowercase hexadecimal characters"
        )
    outputs = preflight_registration(source)
    reference = {
        "schema_version": "1.0",
        "artifact_id": context.artifact_id,
        "kind": kind,
        "relative_uri": context.relative_uri,
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
        "mime_type": mime_type,
        "sensitivity": sensitivity,
        "encryption": encryption,
        "created_at": created_at,
        "parent_artifact_id": None,
        "retention_until": None,
        "producer_run_id": context.run_id,
    }
    manifest = RunManifest(
        run_id=context.run_id,
        experiment_id=context.experiment_id,
        requirement_ids=context.requirement_ids,
        status="completed",
        config_sha256=config_sha256,
        git_commit=git_commit,
        dirty_worktree=dirty_worktree,
        environment_sha256=context.environment_sha256,
        seed=context.seed,
        device=dict(device),
        started_at=started_at,
        ended_at=ended_at,
        input_artifact_ids=context.input_artifact_ids,
        output_artifact_ids=(context.artifact_id,),
        reproduce_command=context.reproduce_command,
        notes=context.notes,
    ).to_dict()

    created: list[Path] = []
    try:
        _write_new_json(outputs.artifact_reference, reference)
        created.append(outputs.artifact_reference)
        _write_new_json(outputs.run_manifest, manifest)
        created.append(outputs.run_manifest)
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return outputs


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(stable_json_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ArtifactRegistrationError(
                f"refusing to overwrite existing registration output: {path}"
            ) from error
    except OSError as error:
        raise ArtifactRegistrationError(
            f"unable to write registration output {path}: {error}"
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactRegistrationError(f"{field} must be a non-empty string")
    return value


def _integer(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ArtifactRegistrationError(f"{field} must be an integer")
    return value


def _string_tuple(
    value: Any, field: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ArtifactRegistrationError(f"{field} must be a JSON string array")
    return tuple(_string(item, field) for item in value)
