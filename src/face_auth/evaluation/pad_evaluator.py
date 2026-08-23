from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from src.common.reproducibility import sha256_file
from src.face_auth.adapters.opencv_capture import video_source
from src.face_auth.domain import reason_codes
from src.face_auth.domain.types import FramePacket, GateStatus
from src.face_auth.evaluation.pad_manifest import PADManifestRow
from src.face_auth.inference.passive_pad import PassivePADGate
from src.face_auth.inference.quality import QualityGate


@dataclass(frozen=True)
class PADSampleResult:
    sample_id: str
    relative_video_path: str
    video_sha256: str | None
    video_bytes: int | None
    subject_token: str
    session_token: str
    device_token: str
    label: str
    attack_species: str
    split: str
    outcome: str
    score: float | None
    threshold: float
    total_frames: int
    valid_face_frames: int
    reason_codes: tuple[str, ...]
    model_version: str
    threshold_version: str
    latency_ms: float

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        return payload


class PADVideoEvaluator:
    def __init__(
        self,
        detector,
        scorer,
        gate: PassivePADGate,
        *,
        quality: QualityGate | None = None,
        max_frames: int = 30,
    ) -> None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        self.detector = detector
        self.scorer = scorer
        self.gate = gate
        self.quality = quality or QualityGate()
        self.max_frames = max_frames

    def evaluate_video(
        self, row: PADManifestRow, video_path: str | Path
    ) -> PADSampleResult:
        started = time.perf_counter()
        try:
            video_sha256 = sha256_file(video_path)
            video_bytes = Path(video_path).stat().st_size
        except OSError:
            return self._result(
                row,
                "ERROR",
                total_frames=0,
                valid_face_frames=0,
                reasons=(reason_codes.CAMERA_ERROR,),
                started=started,
            )
        try:
            frames = _read_frames(video_path, self.max_frames)
        except Exception:
            return self._result(
                row,
                "ERROR",
                total_frames=0,
                valid_face_frames=0,
                reasons=(reason_codes.CAMERA_ERROR,),
                started=started,
                video_sha256=video_sha256,
                video_bytes=video_bytes,
            )
        return self.evaluate_frames(
            row,
            frames,
            started=started,
            video_sha256=video_sha256,
            video_bytes=video_bytes,
        )

    def evaluate_frames(
        self,
        row: PADManifestRow,
        frames: list[FramePacket],
        *,
        started: float | None = None,
        video_sha256: str | None = None,
        video_bytes: int | None = None,
    ) -> PADSampleResult:
        started = time.perf_counter() if started is None else started
        crops = []
        invalid_reasons: Counter[str] = Counter()
        for frame in frames:
            try:
                faces = self.detector.detect(frame.image_bgr)
            except Exception:
                return self._result(
                    row,
                    "ERROR",
                    total_frames=len(frames),
                    valid_face_frames=len(crops),
                    reasons=(reason_codes.MODEL_ERROR,),
                    started=started,
                    video_sha256=video_sha256,
                    video_bytes=video_bytes,
                )
            if len(faces) > 1:
                return self._result(
                    row,
                    "NOT_EVALUATED",
                    total_frames=len(frames),
                    valid_face_frames=len(crops),
                    reasons=(reason_codes.MULTIPLE_FACES,),
                    started=started,
                    video_sha256=video_sha256,
                    video_bytes=video_bytes,
                )
            if not faces:
                invalid_reasons[reason_codes.NO_FACE] += 1
                continue
            try:
                assessment = self.quality.evaluate(
                    frame.image_bgr, faces[0].bbox
                ).result
            except Exception:
                return self._result(
                    row,
                    "ERROR",
                    total_frames=len(frames),
                    valid_face_frames=len(crops),
                    reasons=(reason_codes.MODEL_ERROR,),
                    started=started,
                    video_sha256=video_sha256,
                    video_bytes=video_bytes,
                )
            if assessment.status is not GateStatus.PASS:
                invalid_reasons.update(assessment.reason_codes)
                continue
            crops.append(faces[0].crop)

        try:
            gate_result = self.gate.evaluate(self.scorer.score(crops))
        except Exception:
            return self._result(
                row,
                "ERROR",
                total_frames=len(frames),
                valid_face_frames=len(crops),
                reasons=(reason_codes.MODEL_ERROR,),
                started=started,
                video_sha256=video_sha256,
                video_bytes=video_bytes,
            )

        if gate_result.status is GateStatus.ERROR:
            outcome = "ERROR"
        elif reason_codes.INSUFFICIENT_VALID_FRAMES in gate_result.reason_codes:
            outcome = "NOT_EVALUATED"
        else:
            outcome = gate_result.status.value
        reasons = gate_result.reason_codes
        if outcome == "NOT_EVALUATED" and invalid_reasons:
            reasons = tuple(invalid_reasons.keys()) + tuple(
                code for code in reasons if code not in invalid_reasons
            )
        return self._result(
            row,
            outcome,
            total_frames=len(frames),
            valid_face_frames=len(crops),
            reasons=reasons,
            score=gate_result.score,
            started=started,
            video_sha256=video_sha256,
            video_bytes=video_bytes,
        )

    def _result(
        self,
        row: PADManifestRow,
        outcome: str,
        *,
        total_frames: int,
        valid_face_frames: int,
        reasons: tuple[str, ...],
        started: float,
        score: float | None = None,
        video_sha256: str | None = None,
        video_bytes: int | None = None,
    ) -> PADSampleResult:
        config = self.gate.config
        return PADSampleResult(
            sample_id=row.sample_id,
            relative_video_path=row.relative_video_path,
            video_sha256=video_sha256,
            video_bytes=video_bytes,
            subject_token=row.subject_token,
            session_token=row.session_token,
            device_token=row.device_token,
            label=row.label,
            attack_species=row.attack_species,
            split=row.split,
            outcome=outcome,
            score=score,
            threshold=config.live_threshold,
            total_frames=total_frames,
            valid_face_frames=valid_face_frames,
            reason_codes=reasons,
            model_version=config.model_version,
            threshold_version=config.threshold_version,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )


def evaluate_pad_manifest(
    rows: tuple[PADManifestRow, ...],
    artifact_root: str | Path,
    evaluator: PADVideoEvaluator,
    *,
    split: str = "test",
) -> list[PADSampleResult]:
    root = Path(artifact_root).resolve()
    selected = [row for row in rows if row.split == split]
    if not selected:
        raise ValueError(f"No PAD manifest rows found for split={split}")
    return [
        evaluator.evaluate_video(row, root / row.relative_video_path)
        for row in selected
    ]


def _read_frames(path: str | Path, max_frames: int) -> list[FramePacket]:
    source = video_source(str(path))
    frames: list[FramePacket] = []
    try:
        while len(frames) < max_frames:
            frame = source.read()
            if frame is None:
                break
            frames.append(frame)
    finally:
        source.close()
    return frames
