"""Immutable run-manifest representation for new experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    experiment_id: str
    requirement_ids: tuple[str, ...]
    status: str
    config_sha256: str
    git_commit: str
    environment_sha256: str
    seed: int
    device: dict[str, Any]
    started_at: str
    input_artifact_ids: tuple[str, ...]
    output_artifact_ids: tuple[str, ...]
    reproduce_command: str
    schema_version: str = "1.0"
    dirty_worktree: bool = False
    ended_at: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.status == "completed" and self.ended_at is None:
            raise ValueError("completed run requires ended_at")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.requirement_ids:
            raise ValueError("at least one requirement ID is required")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["requirement_ids"] = list(self.requirement_ids)
        result["input_artifact_ids"] = list(self.input_artifact_ids)
        result["output_artifact_ids"] = list(self.output_artifact_ids)
        return result

    @staticmethod
    def utc_now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
