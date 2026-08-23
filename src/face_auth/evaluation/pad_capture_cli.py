from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.face_auth.adapters.opencv_capture import webcam_source
from src.face_auth.evaluation.pad_capture import (
    PADCaptureConfig,
    PADCaptureRecorder,
    relative_pad_video_path,
)
from src.face_auth.evaluation.pad_manifest import (
    ATTACK_SPECIES,
    LABELS,
    SPLITS,
    PADManifestRow,
    validate_pad_manifest,
)


def main() -> int:
    args = _parser().parse_args()
    _validate_args(args)
    provisional = PADManifestRow(
        sample_id=args.sample_id,
        label=args.label,
        attack_species=args.attack_species,
        relative_video_path="placeholder.mp4",
        subject_token=args.subject_token,
        session_token=args.session_token,
        device_token=args.device_token,
        split=args.split,
    )
    row = PADManifestRow(
        **{
            **provisional.__dict__,
            "relative_video_path": relative_pad_video_path(provisional),
        }
    )
    try:
        validate_pad_manifest([row])
    except ValueError as error:
        raise SystemExit(str(error)) from error
    for remaining in range(args.countdown, 0, -1):
        print(f"capture starts in {remaining}", flush=True)
        time.sleep(1)
    try:
        source = webcam_source(args.camera)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    recorder = PADCaptureRecorder(source)
    try:
        receipt = recorder.capture(
            row,
            PADCaptureConfig(
                artifact_root=Path(args.artifact_root),
                manifest_path=Path(args.manifest),
                fps=args.fps,
                frame_count=round(args.duration_seconds * args.fps),
                min_frames=args.min_frames,
                width=args.width,
                height=args.height,
                require_device_disjoint=args.require_device_disjoint,
            ),
        )
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(receipt.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record one pseudonymous physical PAD evaluation clip"
    )
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--label", choices=sorted(LABELS), required=True)
    parser.add_argument(
        "--attack-species", choices=sorted(ATTACK_SPECIES), required=True
    )
    parser.add_argument("--subject-token", required=True)
    parser.add_argument("--session-token", required=True)
    parser.add_argument("--device-token", required=True)
    parser.add_argument("--split", choices=sorted(SPLITS), required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duration-seconds", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--require-device-disjoint", action="store_true")
    return parser


def _validate_args(args) -> None:
    if args.duration_seconds <= 0 or args.fps <= 0:
        raise SystemExit("--duration-seconds and --fps must be positive")
    if round(args.duration_seconds * args.fps) < args.min_frames:
        raise SystemExit("capture duration is too short for --min-frames")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive")
    if args.countdown < 0:
        raise SystemExit("--countdown must not be negative")


if __name__ == "__main__":
    raise SystemExit(main())
