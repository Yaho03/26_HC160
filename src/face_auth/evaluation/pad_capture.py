from __future__ import annotations

import csv
import os
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2

from src.common.reproducibility import sha256_file
from src.face_auth.evaluation.pad_manifest import (
    PADManifestRow,
    REQUIRED_COLUMNS,
    load_pad_manifest,
    validate_pad_manifest,
)


@dataclass(frozen=True)
class PADCaptureConfig:
    artifact_root: Path
    manifest_path: Path
    fps: float = 15.0
    frame_count: int = 75
    min_frames: int = 30
    width: int = 640
    height: int = 480
    require_device_disjoint: bool = False


@dataclass(frozen=True)
class PADCaptureReceipt:
    sample_id: str
    relative_video_path: str
    video_sha256: str
    video_bytes: int
    frame_count: int
    fps: float
    width: int
    height: int

    def to_dict(self) -> dict:
        return asdict(self)


class PADCaptureRecorder:
    """Record one labeled PAD clip and atomically register its manifest row."""

    def __init__(
        self,
        source,
        *,
        writer_factory: Callable = cv2.VideoWriter,
        resize: Callable = cv2.resize,
    ) -> None:
        self.source = source
        self.writer_factory = writer_factory
        self.resize = resize

    def capture(
        self, row: PADManifestRow, config: PADCaptureConfig
    ) -> PADCaptureReceipt:
        try:
            root = config.artifact_root.resolve()
            root.mkdir(parents=True, exist_ok=True)
            with _exclusive_capture_lock(root):
                return self._capture(row, config)
        finally:
            self.source.close()

    def _capture(
        self, row: PADManifestRow, config: PADCaptureConfig
    ) -> PADCaptureReceipt:
        _validate_capture_config(config)
        root = config.artifact_root.resolve()
        manifest = _inside_root(root, config.manifest_path, "manifest")
        target = _inside_root(root, root / row.relative_video_path, "video")
        existing_rows, manifest_snapshot = _load_manifest_snapshot(manifest)
        candidate_rows = (*existing_rows, row)
        validate_pad_manifest(
            candidate_rows,
            require_device_disjoint=config.require_device_disjoint,
        )
        if target.exists():
            raise ValueError(f"Refusing to overwrite existing PAD video: {target}")

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(
            f".{target.stem}.{uuid.uuid4().hex}.partial{target.suffix}"
        )
        writer = None
        captured = 0
        try:
            try:
                writer = self.writer_factory(
                    str(temporary),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    config.fps,
                    (config.width, config.height),
                )
                if hasattr(writer, "isOpened") and not writer.isOpened():
                    raise RuntimeError("Cannot open PAD video writer")
                while captured < config.frame_count:
                    packet = self.source.read()
                    if packet is None:
                        break
                    frame = self.resize(packet.image_bgr, (config.width, config.height))
                    writer.write(frame)
                    captured += 1
            finally:
                if writer is not None:
                    writer.release()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

        if captured < config.min_frames:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"PAD capture has too few frames: {captured} < {config.min_frames}"
            )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("PAD video writer produced no artifact")

        os.replace(temporary, target)
        try:
            if _read_optional_bytes(manifest) != manifest_snapshot:
                raise RuntimeError("PAD manifest changed while capture was in progress")
            validate_pad_manifest(
                candidate_rows,
                artifact_root=root,
                require_device_disjoint=config.require_device_disjoint,
            )
            _atomic_write_manifest(manifest, candidate_rows)
        except BaseException:
            target.unlink(missing_ok=True)
            raise

        return PADCaptureReceipt(
            sample_id=row.sample_id,
            relative_video_path=row.relative_video_path,
            video_sha256=sha256_file(target),
            video_bytes=target.stat().st_size,
            frame_count=captured,
            fps=config.fps,
            width=config.width,
            height=config.height,
        )


def relative_pad_video_path(row: PADManifestRow) -> str:
    category = "bona_fide" if row.label == "bona_fide" else row.attack_species
    return f"{row.split}/{category}/{row.sample_id}.mp4"


def _validate_capture_config(config: PADCaptureConfig) -> None:
    if config.fps <= 0:
        raise ValueError("PAD capture fps must be positive")
    if config.min_frames <= 0 or config.frame_count < config.min_frames:
        raise ValueError("PAD capture requires 0 < min_frames <= frame_count")
    if config.width <= 0 or config.height <= 0:
        raise ValueError("PAD capture dimensions must be positive")


def _load_manifest_snapshot(
    manifest: Path,
) -> tuple[tuple[PADManifestRow, ...], bytes | None]:
    snapshot = _read_optional_bytes(manifest)
    rows = load_pad_manifest(manifest) if snapshot is not None else ()
    return rows, snapshot


def _atomic_write_manifest(path: Path, rows: tuple[PADManifestRow, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _inside_root(root: Path, value: Path, kind: str) -> Path:
    candidate = value.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"PAD {kind} path must remain inside artifact root") from error
    return candidate


def _read_optional_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


@contextmanager
def _exclusive_capture_lock(root: Path):
    lock_path = root / ".pad-capture.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise RuntimeError(
            f"Another PAD capture owns the dataset lock: {lock_path}"
        ) from error
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)
