from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.reproducibility import (
    git_state,
    sha256_file,
    stable_json_bytes,
    stable_json_sha256,
)
from src.experiments.artifact_registration import (
    ArtifactRegistrationError,
    RegistrationContext,
    load_registration_context,
    preflight_registration,
    register_completed_output,
)
from src.experiments.run_manifest import RunManifest
from src.face_auth.evaluation.pad_evaluator import (
    PADVideoEvaluator,
    evaluate_pad_manifest,
)
from src.face_auth.evaluation.pad_manifest import (
    load_pad_manifest,
    validate_pad_manifest,
)
from src.face_auth.evaluation.pad_metrics import (
    calibrate_pad_threshold,
    pad_metrics,
    reclassify_pad_samples,
)
from src.face_auth.inference.face_detector import MTCNNFaceDetector
from src.face_auth.inference.pad_adapter import create_pad_scorer
from src.face_auth.inference.passive_pad import PassivePADConfig, PassivePADGate
from src.face_auth.inference.quality import QualityConfig, QualityGate


def main() -> int:
    args = _parser().parse_args()
    _validate_args(args)
    target = Path(args.output)
    context, registration_started_at = _prepare_registration(args, target)
    protected_inputs = [Path(args.manifest), Path(args.pad_model)]
    if args.registration_context is not None:
        protected_inputs.append(Path(args.registration_context))
    _validate_output_path(
        target,
        inputs=tuple(protected_inputs),
        overwrite=args.overwrite,
    )
    try:
        code_state = git_state(args.repo_dir)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"unable to identify repository state: {exc}") from exc
    if code_state["dirty_worktree"] and not args.allow_dirty:
        raise SystemExit(
            "refusing a report from a dirty worktree; commit changes or pass --allow-dirty"
        )
    manifest_sha256 = sha256_file(args.manifest)
    model_sha256 = sha256_file(args.pad_model)
    evaluation_threshold = (
        args.live_threshold if args.live_threshold is not None else 0.5
    )
    rows = load_pad_manifest(args.manifest)
    validate_pad_manifest(
        rows,
        artifact_root=args.artifact_root,
        require_device_disjoint=args.require_device_disjoint,
    )
    scorer = create_pad_scorer(
        runtime=args.pad_runtime,
        model_path=args.pad_model,
        model_version=args.pad_model_version,
        input_size=args.pad_input_size,
        live_class_index=args.pad_live_class_index,
        output_kind=args.pad_output_kind,
        device=args.device,
        providers=args.pad_provider,
    )
    evaluator = PADVideoEvaluator(
        MTCNNFaceDetector(device=args.device),
        scorer,
        PassivePADGate(
            PassivePADConfig(
                live_threshold=evaluation_threshold,
                model_version=args.pad_model_version,
                threshold_version=args.threshold_version,
                min_frames=args.min_valid_frames,
            )
        ),
        quality=QualityGate(
            QualityConfig(
                min_blur_variance=args.min_blur_variance,
                min_mean_brightness=args.min_brightness,
                max_mean_brightness=args.max_brightness,
            )
        ),
        max_frames=args.max_frames,
    )
    results = evaluate_pad_manifest(
        rows, args.artifact_root, evaluator, split=args.split
    )
    sample_payloads = [result.to_dict() for result in results]
    calibration = None
    reported_threshold = evaluation_threshold
    if args.mode == "calibrate":
        candidate = calibrate_pad_threshold(sample_payloads, max_bpcer=args.max_bpcer)
        calibration = candidate.to_dict()
        reported_threshold = candidate.threshold
        sample_payloads = reclassify_pad_samples(sample_payloads, reported_threshold)
    _verify_inputs_unchanged(
        manifest_path=Path(args.manifest),
        manifest_sha256=manifest_sha256,
        model_path=Path(args.pad_model),
        model_sha256=model_sha256,
        artifact_root=Path(args.artifact_root),
        samples=sample_payloads,
    )
    if git_state(args.repo_dir) != code_state:
        raise SystemExit("repository state changed during PAD evaluation")
    payload = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code": code_state,
        "manifest": {
            "id": args.manifest_id,
            "sha256": manifest_sha256,
            "row_count": len(rows),
            "selected_row_count": len(results),
        },
        "mode": args.mode,
        "split": args.split,
        "model": {**scorer.metadata(), "sha256": model_sha256},
        "threshold": {
            "value": reported_threshold,
            "version": args.threshold_version,
            "max_bpcer": args.max_bpcer if args.mode == "calibrate" else None,
        },
        "calibration": calibration,
        "evaluation_config": {
            "max_frames": args.max_frames,
            "min_valid_frames": args.min_valid_frames,
            "quality": {
                "min_blur_variance": args.min_blur_variance,
                "min_brightness": args.min_brightness,
                "max_brightness": args.max_brightness,
            },
            "device": args.device,
            "require_device_disjoint": args.require_device_disjoint,
        },
        "metrics": pad_metrics(sample_payloads),
        "samples": sample_payloads,
    }
    _atomic_json_write(target, payload, overwrite=args.overwrite)
    if context is not None:
        try:
            register_completed_output(
                target,
                context=context,
                kind="report",
                created_at=payload["created_at"],
                config_sha256=_pad_config_sha256(args, model_sha256),
                git_commit=code_state["git_commit"],
                dirty_worktree=code_state["dirty_worktree"],
                device={
                    "type": args.device,
                    "runtime": args.pad_runtime,
                    "providers": args.pad_provider or [],
                },
                started_at=registration_started_at,
                ended_at=RunManifest.utc_now(),
            )
        except (ArtifactRegistrationError, OSError) as error:
            target.unlink(missing_ok=True)
            raise SystemExit(f"unable to register PAD report: {error}") from error
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))
    return 0 if payload["metrics"]["sample_counts"]["evaluated"] else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate passive PAD on labeled videos"
    )
    parser.add_argument("--mode", choices=["calibrate", "evaluate"], default="evaluate")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-id", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--pad-model", required=True)
    parser.add_argument("--pad-model-version", required=True)
    parser.add_argument(
        "--pad-runtime", choices=["torchscript", "onnx"], default="torchscript"
    )
    parser.add_argument(
        "--pad-provider",
        action="append",
        help="ONNX Runtime execution provider; repeat to set fallback order",
    )
    parser.add_argument("--live-threshold", type=float)
    parser.add_argument("--threshold-version", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--registration-context",
        help=(
            "write immutable artifact-reference and run-manifest sidecars from "
            "an explicit registration context"
        ),
    )
    parser.add_argument("--split", choices=["calibration", "test"], default="test")
    parser.add_argument("--max-bpcer", type=float, default=0.05)
    parser.add_argument("--require-device-disjoint", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--min-valid-frames", type=int, default=5)
    parser.add_argument("--pad-input-size", type=int)
    parser.add_argument("--pad-live-class-index", type=int)
    parser.add_argument(
        "--pad-output-kind", choices=["logits", "probability"]
    )
    parser.add_argument("--min-blur-variance", type=float, default=40.0)
    parser.add_argument("--min-brightness", type=float, default=35.0)
    parser.add_argument("--max-brightness", type=float, default=220.0)
    return parser


def _validate_args(args) -> None:
    if args.mode == "evaluate" and args.live_threshold is None:
        raise SystemExit("evaluate mode requires --live-threshold")
    if args.mode == "calibrate" and args.split != "calibration":
        raise SystemExit("calibrate mode requires --split calibration")
    if args.live_threshold is not None and not 0.0 <= args.live_threshold <= 1.0:
        raise SystemExit("--live-threshold must be in [0, 1]")
    if not 0.0 <= args.max_bpcer < 1.0:
        raise SystemExit("--max-bpcer must be in [0, 1)")
    if args.min_valid_frames <= 0 or args.max_frames < args.min_valid_frames:
        raise SystemExit("frame limits require 0 < min-valid-frames <= max-frames")
    if args.pad_input_size is not None and args.pad_input_size <= 0:
        raise SystemExit("--pad-input-size must be positive")
    if args.pad_live_class_index is not None and args.pad_live_class_index < 0:
        raise SystemExit("--pad-live-class-index must not be negative")
    if not args.run_id.strip():
        raise SystemExit("--run-id must not be empty")
    if not args.manifest_id.strip():
        raise SystemExit("--manifest-id must not be empty")


def _prepare_registration(
    args, target: Path
) -> tuple[RegistrationContext | None, str | None]:
    if args.registration_context is None:
        return None, None
    if args.overwrite:
        raise SystemExit(
            "registered PAD outputs are immutable; remove --overwrite"
        )
    try:
        context = load_registration_context(args.registration_context)
        preflight_registration(target)
    except ArtifactRegistrationError as error:
        raise SystemExit(f"unable to prepare PAD registration: {error}") from error
    if context.run_id != args.run_id:
        raise SystemExit(
            "registration context run_id must match the PAD --run-id"
        )
    return context, RunManifest.utc_now()


def _pad_config_sha256(args, model_sha256: str) -> str:
    return stable_json_sha256(
        {
            "mode": args.mode,
            "split": args.split,
            "manifest_id": args.manifest_id,
            "pad_model": {
                "version": args.pad_model_version,
                "sha256": model_sha256,
                "runtime": args.pad_runtime,
                "providers": args.pad_provider or [],
                "input_size": args.pad_input_size,
                "live_class_index": args.pad_live_class_index,
                "output_kind": args.pad_output_kind,
            },
            "live_threshold": args.live_threshold,
            "threshold_version": args.threshold_version,
            "max_bpcer": args.max_bpcer,
            "device": args.device,
            "max_frames": args.max_frames,
            "min_valid_frames": args.min_valid_frames,
            "require_device_disjoint": args.require_device_disjoint,
            "quality": {
                "min_blur_variance": args.min_blur_variance,
                "min_brightness": args.min_brightness,
                "max_brightness": args.max_brightness,
            },
        }
    )


def _validate_output_path(
    output: Path, *, inputs: tuple[Path, ...], overwrite: bool
) -> None:
    resolved_output = output.resolve()
    if resolved_output in {path.resolve() for path in inputs}:
        raise SystemExit("--output must differ from all input files")
    if output.exists() and not overwrite:
        raise SystemExit(f"refusing to overwrite existing file: {output}")


def _atomic_json_write(path: Path, value: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = stable_json_bytes(value) + b"\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if overwrite:
            os.replace(temporary, path)
            temporary = None
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise SystemExit(
                    f"refusing to overwrite existing file: {path}"
                ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _verify_inputs_unchanged(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    model_path: Path,
    model_sha256: str,
    artifact_root: Path,
    samples: list[dict[str, Any]],
) -> None:
    try:
        if sha256_file(manifest_path) != manifest_sha256:
            raise SystemExit("PAD manifest changed during evaluation")
        if sha256_file(model_path) != model_sha256:
            raise SystemExit("PAD model changed during evaluation")
        for sample in samples:
            expected_hash = sample.get("video_sha256")
            expected_bytes = sample.get("video_bytes")
            if expected_hash is None or expected_bytes is None:
                raise SystemExit(
                    "source video provenance is unavailable: "
                    f"{sample['relative_video_path']}"
                )
            path = artifact_root / sample["relative_video_path"]
            if (
                path.stat().st_size != expected_bytes
                or sha256_file(path) != expected_hash
            ):
                raise SystemExit(
                    "source video changed during evaluation: "
                    f"{sample['relative_video_path']}"
                )
    except OSError as exc:
        raise SystemExit(f"unable to re-verify PAD inputs: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
