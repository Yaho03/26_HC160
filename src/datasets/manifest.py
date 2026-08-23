"""Build and validate privacy-conscious dataset manifests.

The builder accepts only artifact-ready datasets whose identity directories
already use pseudonymous tokens. It never writes raw identity names to a
manifest and requires no image-processing dependency.
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from src.common.reproducibility import sha256_file, stable_json_bytes
from src.contracts.validation import ContractError, validate_relative_uri, validate_sha256


ALLOWED_SPLITS = ("train", "calibration", "development", "test")
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}
IDENTITY_PREFIX = "id_"


class DatasetManifestError(ValueError):
    """Raised when dataset discovery or semantic validation fails."""


@dataclass(frozen=True)
class DatasetManifestRow:
    schema_version: str
    dataset_id: str
    sample_id: str
    identity_token: str
    relative_uri: str
    media_sha256: str
    split: str
    width_px: int
    height_px: int
    license_id: str
    source_uri: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.source_uri is None:
            result.pop("source_uri")
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DatasetManifestRow":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise DatasetManifestError(f"unknown manifest fields: {', '.join(unknown)}")
        try:
            return cls(**value)
        except TypeError as exc:
            raise DatasetManifestError(f"invalid manifest row: {exc}") from exc


@dataclass(frozen=True)
class ManifestValidationReport:
    dataset_id: str
    license_id: str
    row_count: int
    identity_count: int
    split_counts: dict[str, int]
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_manifest_rows(
    artifact_root: str | Path,
    dataset_id: str,
    license_id: str,
    source_uri: str | None = None,
) -> tuple[DatasetManifestRow, ...]:
    """Discover ``<split>/<identity_token>/<image>`` files deterministically."""
    root = Path(artifact_root)
    if not root.is_dir():
        raise DatasetManifestError(f"artifact root is not a directory: {root}")
    if not dataset_id.strip() or not license_id.strip():
        raise DatasetManifestError("dataset_id and license_id must be non-empty")

    rows: list[DatasetManifestRow] = []
    for split in ALLOWED_SPLITS:
        split_root = root / split
        if not split_root.exists():
            continue
        if not split_root.is_dir():
            raise DatasetManifestError(f"split path is not a directory: {split_root}")

        files = sorted(
            path
            for path in split_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        for path in files:
            _ensure_within_root(path, root)
            within_split = path.relative_to(split_root)
            if len(within_split.parts) != 2:
                raise DatasetManifestError(
                    f"image path must be exactly <split>/<identity-token>/<file>: {path}"
                )
            identity_token = within_split.parts[0]
            _validate_identity_token(identity_token)
            width, height = image_dimensions(path)
            media_hash = sha256_file(path)
            rows.append(
                DatasetManifestRow(
                    schema_version="1.0",
                    dataset_id=dataset_id,
                    sample_id=f"sample_{media_hash[:24]}",
                    identity_token=identity_token,
                    relative_uri=path.relative_to(root).as_posix(),
                    media_sha256=media_hash,
                    split=split,
                    width_px=width,
                    height_px=height,
                    license_id=license_id,
                    source_uri=source_uri,
                )
            )
    if not rows:
        raise DatasetManifestError("no PNG or JPEG images found in supported split directories")
    return tuple(sorted(rows, key=lambda row: (row.relative_uri, row.sample_id)))


def validate_manifest_rows(
    rows: Iterable[DatasetManifestRow],
    artifact_root: str | Path | None = None,
    require_identity_disjoint: bool = False,
) -> ManifestValidationReport:
    materialized = tuple(rows)
    if not materialized:
        raise DatasetManifestError("manifest must contain at least one row")

    sample_ids: set[str] = set()
    relative_uris: set[str] = set()
    hash_split: dict[str, str] = {}
    identity_split: dict[str, str] = {}
    dataset_ids: set[str] = set()
    license_ids: set[str] = set()
    split_counts = {split: 0 for split in ALLOWED_SPLITS}
    root = Path(artifact_root).resolve() if artifact_root is not None else None

    for row in materialized:
        _validate_row_types(row)
        try:
            validate_relative_uri(row.relative_uri)
            validate_sha256(row.media_sha256)
        except ContractError as exc:
            raise DatasetManifestError(str(exc)) from exc
        _validate_identity_token(row.identity_token)

        if row.sample_id in sample_ids:
            raise DatasetManifestError(f"duplicate sample_id: {row.sample_id}")
        if row.relative_uri in relative_uris:
            raise DatasetManifestError(f"duplicate relative_uri: {row.relative_uri}")
        sample_ids.add(row.sample_id)
        relative_uris.add(row.relative_uri)
        dataset_ids.add(row.dataset_id)
        license_ids.add(row.license_id)
        split_counts[row.split] += 1

        previous_split = hash_split.setdefault(row.media_sha256, row.split)
        if previous_split != row.split:
            raise DatasetManifestError(
                f"media hash crosses splits: {row.media_sha256} "
                f"({previous_split}, {row.split})"
            )

        if require_identity_disjoint:
            previous_identity_split = identity_split.setdefault(row.identity_token, row.split)
            if previous_identity_split != row.split:
                raise DatasetManifestError(
                    f"identity crosses splits: {row.identity_token} "
                    f"({previous_identity_split}, {row.split})"
                )

        if root is not None:
            _validate_referenced_file(row, root)

    if len(dataset_ids) != 1:
        raise DatasetManifestError("all rows must use one dataset_id")
    if len(license_ids) != 1:
        raise DatasetManifestError("all rows must use one license_id")

    return ManifestValidationReport(
        dataset_id=next(iter(dataset_ids)),
        license_id=next(iter(license_ids)),
        row_count=len(materialized),
        identity_count=len({row.identity_token for row in materialized}),
        split_counts=split_counts,
        manifest_sha256=sha256(canonical_manifest_bytes(materialized)).hexdigest(),
    )


def canonical_manifest_bytes(rows: Iterable[DatasetManifestRow]) -> bytes:
    ordered = sorted(rows, key=lambda row: (row.relative_uri, row.sample_id))
    return b"".join(stable_json_bytes(row.to_dict()) + b"\n" for row in ordered)


def load_manifest(path: str | Path) -> tuple[DatasetManifestRow, ...]:
    rows: list[DatasetManifestRow] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetManifestError(
                    f"invalid JSON on manifest line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise DatasetManifestError(f"manifest line {line_number} must be an object")
            rows.append(DatasetManifestRow.from_mapping(value))
    return tuple(rows)


def write_manifest(
    rows: Iterable[DatasetManifestRow],
    output_path: str | Path,
    overwrite: bool = False,
) -> str:
    output = Path(output_path)
    payload = canonical_manifest_bytes(rows)
    _atomic_write(output, payload, overwrite)
    return sha256(payload).hexdigest()


def build_snapshot_metadata(
    *,
    dataset_id: str,
    manifest_relative_uri: str,
    report: ManifestValidationReport,
    source_archive_sha256: str,
    source_uri: str,
    source_retrieved_at: str,
    license_id: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    if dataset_id != report.dataset_id:
        raise DatasetManifestError("snapshot dataset_id differs from manifest dataset_id")
    if license_id != report.license_id:
        raise DatasetManifestError("snapshot license_id differs from manifest license_id")
    if not isinstance(source_archive_sha256, str):
        raise DatasetManifestError("source_archive_sha256 must be a string")
    try:
        validate_relative_uri(manifest_relative_uri)
        validate_sha256(source_archive_sha256)
    except ContractError as exc:
        raise DatasetManifestError(str(exc)) from exc
    if not source_uri.strip() or not license_id.strip():
        raise DatasetManifestError("source_uri and license_id must be non-empty")
    parsed_source = urlparse(source_uri)
    if not parsed_source.scheme or not (parsed_source.netloc or parsed_source.path):
        raise DatasetManifestError("source_uri must be an absolute URI")
    try:
        date.fromisoformat(source_retrieved_at)
    except ValueError as exc:
        raise DatasetManifestError("source_retrieved_at must use YYYY-MM-DD") from exc
    timestamp = created_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DatasetManifestError("created_at must be an ISO 8601 timestamp") from exc
    return {
        "schema_version": "1.0",
        "snapshot_id": f"snapshot_{dataset_id}_{report.manifest_sha256[:12]}",
        "dataset_id": dataset_id,
        "manifest_relative_uri": manifest_relative_uri,
        "manifest_sha256": report.manifest_sha256,
        "source_archive_sha256": source_archive_sha256,
        "source_uri": source_uri,
        "source_retrieved_at": source_retrieved_at,
        "license_id": license_id,
        "identity_token_policy": "pre-pseudonymized-id_-hex-v1",
        "row_count": report.row_count,
        "identity_count": report.identity_count,
        "split_counts": report.split_counts,
        "created_at": timestamp,
    }


def write_snapshot_metadata(
    metadata: Mapping[str, Any],
    output_path: str | Path,
    overwrite: bool = False,
) -> None:
    _atomic_write(Path(output_path), stable_json_bytes(metadata) + b"\n", overwrite)


def image_dimensions(path: str | Path) -> tuple[int, int]:
    image_path = Path(path)
    suffix = image_path.suffix.lower()
    with image_path.open("rb") as handle:
        if suffix == ".png":
            header = handle.read(24)
            if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
                raise DatasetManifestError(f"invalid PNG header: {image_path}")
            width, height = struct.unpack(">II", header[16:24])
        elif suffix in {".jpg", ".jpeg"}:
            width, height = _jpeg_dimensions(handle, image_path)
        else:
            raise DatasetManifestError(f"unsupported image format: {image_path}")
    if width <= 0 or height <= 0:
        raise DatasetManifestError(f"invalid image dimensions: {image_path}")
    return width, height


def _jpeg_dimensions(handle: Any, path: Path) -> tuple[int, int]:
    if handle.read(2) != b"\xff\xd8":
        raise DatasetManifestError(f"invalid JPEG header: {path}")
    start_of_frame = {
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
    }
    while True:
        marker_prefix = handle.read(1)
        if not marker_prefix:
            break
        if marker_prefix != b"\xff":
            continue
        marker = handle.read(1)
        while marker == b"\xff":
            marker = handle.read(1)
        if not marker:
            break
        marker_value = marker[0]
        if marker_value in {0x01, *range(0xD0, 0xD9)}:
            continue
        length_bytes = handle.read(2)
        if len(length_bytes) != 2:
            break
        segment_length = int.from_bytes(length_bytes, "big")
        if segment_length < 2:
            break
        if marker_value in start_of_frame:
            segment = handle.read(segment_length - 2)
            if len(segment) < 5:
                break
            return int.from_bytes(segment[3:5], "big"), int.from_bytes(segment[1:3], "big")
        handle.seek(segment_length - 2, os.SEEK_CUR)
    raise DatasetManifestError(f"JPEG dimensions not found: {path}")


def _validate_row_types(row: DatasetManifestRow) -> None:
    if row.split not in ALLOWED_SPLITS:
        raise DatasetManifestError(f"unsupported split: {row.split}")
    for field_name in ("dataset_id", "sample_id", "identity_token", "license_id"):
        value = getattr(row, field_name)
        if not isinstance(value, str) or not value.strip():
            raise DatasetManifestError(f"{field_name} must be a non-empty string")
    if type(row.width_px) is not int or type(row.height_px) is not int:
        raise DatasetManifestError("image dimensions must be integers")
    if row.width_px <= 0 or row.height_px <= 0:
        raise DatasetManifestError("image dimensions must be positive")


def _validate_identity_token(value: str) -> None:
    suffix = value[len(IDENTITY_PREFIX):] if value.startswith(IDENTITY_PREFIX) else ""
    if len(suffix) < 16 or any(char not in "0123456789abcdef" for char in suffix):
        raise DatasetManifestError(
            "identity directories must be pseudonymous tokens matching id_<16+ lowercase hex>"
        )


def _validate_referenced_file(row: DatasetManifestRow, root: Path) -> None:
    candidate = (root / row.relative_uri).resolve()
    _ensure_within_root(candidate, root)
    if not candidate.is_file():
        raise DatasetManifestError(f"referenced artifact does not exist: {row.relative_uri}")
    if sha256_file(candidate) != row.media_sha256:
        raise DatasetManifestError(f"artifact hash mismatch: {row.relative_uri}")
    width, height = image_dimensions(candidate)
    if (width, height) != (row.width_px, row.height_px):
        raise DatasetManifestError(f"artifact dimensions changed: {row.relative_uri}")


def _ensure_within_root(candidate: Path, root: Path) -> None:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise DatasetManifestError(f"artifact escapes root: {candidate}") from exc


def _atomic_write(path: Path, payload: bytes, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise DatasetManifestError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
