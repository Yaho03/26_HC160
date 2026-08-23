from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


LABELS = frozenset({"bona_fide", "attack"})
ATTACK_SPECIES = frozenset(
    {
        "none",
        "print",
        "screen_static",
        "screen_replay",
        "adversarial_print",
        "adversarial_screen",
    }
)
SPLITS = frozenset({"calibration", "test"})
REQUIRED_COLUMNS = (
    "sample_id",
    "label",
    "attack_species",
    "relative_video_path",
    "subject_token",
    "session_token",
    "device_token",
    "split",
)
_TOKEN = re.compile(r"^[a-z][a-z0-9_-]{7,}$")


class PADManifestError(ValueError):
    pass


@dataclass(frozen=True)
class PADManifestRow:
    sample_id: str
    label: str
    attack_species: str
    relative_video_path: str
    subject_token: str
    session_token: str
    device_token: str
    split: str


def load_pad_manifest(path: str | Path) -> tuple[PADManifestRow, ...]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in (reader.fieldnames or [])
        ]
        if missing:
            raise PADManifestError(
                f"Missing PAD manifest columns: {', '.join(missing)}"
            )
        unknown = sorted(set(reader.fieldnames or []) - set(REQUIRED_COLUMNS))
        if unknown:
            raise PADManifestError(
                f"Unknown PAD manifest columns: {', '.join(unknown)}"
            )
        rows = tuple(
            PADManifestRow(**{column: record[column] for column in REQUIRED_COLUMNS})
            for record in reader
        )
    validate_pad_manifest(rows)
    return rows


def validate_pad_manifest(
    rows: tuple[PADManifestRow, ...] | list[PADManifestRow],
    *,
    artifact_root: str | Path | None = None,
    require_subject_disjoint: bool = True,
    require_device_disjoint: bool = False,
) -> None:
    if not rows:
        raise PADManifestError("PAD manifest must contain at least one row")
    sample_ids: set[str] = set()
    video_paths: set[str] = set()
    session_tokens: set[str] = set()
    subject_splits: dict[str, str] = {}
    device_splits: dict[str, str] = {}
    root = Path(artifact_root).resolve() if artifact_root is not None else None
    for row in rows:
        if row.sample_id in sample_ids:
            raise PADManifestError(f"Duplicate sample_id: {row.sample_id}")
        if row.relative_video_path in video_paths:
            raise PADManifestError(
                f"Duplicate relative_video_path: {row.relative_video_path}"
            )
        sample_ids.add(row.sample_id)
        video_paths.add(row.relative_video_path)
        if row.session_token in session_tokens:
            raise PADManifestError(f"Duplicate session_token: {row.session_token}")
        session_tokens.add(row.session_token)

        if not _TOKEN.fullmatch(row.sample_id):
            raise PADManifestError(
                f"sample_id must be an opaque token: {row.sample_id}"
            )
        for field in ("subject_token", "session_token", "device_token"):
            value = getattr(row, field)
            if not _TOKEN.fullmatch(value):
                raise PADManifestError(f"{field} must be an opaque token: {value}")
        if row.label not in LABELS:
            raise PADManifestError(f"Unsupported label: {row.label}")
        if row.attack_species not in ATTACK_SPECIES:
            raise PADManifestError(f"Unsupported attack_species: {row.attack_species}")
        if row.split not in SPLITS:
            raise PADManifestError(f"Unsupported split: {row.split}")
        if row.label == "bona_fide" and row.attack_species != "none":
            raise PADManifestError("bona_fide rows require attack_species=none")
        if row.label == "attack" and row.attack_species == "none":
            raise PADManifestError("attack rows require a concrete attack_species")
        if require_subject_disjoint:
            _require_one_split(subject_splits, row.subject_token, row.split, "subject")
        if require_device_disjoint:
            _require_one_split(device_splits, row.device_token, row.split, "device")
        _validate_relative_video_path(row.relative_video_path)

        if root is not None:
            candidate = (root / row.relative_video_path).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise PADManifestError(
                    f"Video path escapes artifact root: {row.relative_video_path}"
                ) from exc
            if not candidate.is_file():
                raise PADManifestError(
                    f"Video does not exist: {row.relative_video_path}"
                )


def _validate_relative_video_path(value: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise PADManifestError(
            f"Video path must be a safe relative POSIX path: {value}"
        )
    if path.suffix.lower() not in {".mp4", ".mov", ".avi"}:
        raise PADManifestError(f"Unsupported video extension: {value}")


def _require_one_split(
    assignments: dict[str, str], token: str, split: str, kind: str
) -> None:
    previous = assignments.setdefault(token, split)
    if previous != split:
        raise PADManifestError(
            f"{kind} token crosses splits: {token} ({previous}, {split})"
        )
