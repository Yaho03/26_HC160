from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

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
from src.face_auth.inference.pad_adapter import (
    TorchScriptPADConfig,
    TorchScriptPADScorer,
)
from src.face_auth.inference.passive_pad import PassivePADConfig, PassivePADGate
from src.face_auth.inference.quality import QualityConfig, QualityGate


def main() -> int:
    args = _parser().parse_args()
    _validate_args(args)
    evaluation_threshold = (
        args.live_threshold if args.live_threshold is not None else 0.5
    )
    rows = load_pad_manifest(args.manifest)
    validate_pad_manifest(
        rows,
        artifact_root=args.artifact_root,
        require_device_disjoint=args.require_device_disjoint,
    )
    scorer = TorchScriptPADScorer(
        TorchScriptPADConfig(
            model_path=args.pad_model,
            model_version=args.pad_model_version,
            input_size=args.pad_input_size,
            live_class_index=args.pad_live_class_index,
            output_kind=args.pad_output_kind,
        ),
        device=args.device,
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
    payload = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(Path(args.manifest)),
        "mode": args.mode,
        "split": args.split,
        "model": {
            "version": args.pad_model_version,
            "sha256": _sha256(args.pad_model),
            "input_size": args.pad_input_size,
            "output_kind": args.pad_output_kind,
            "live_class_index": args.pad_live_class_index,
            "mean": list(scorer.config.mean),
            "std": list(scorer.config.std),
        },
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
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))
    return 0 if payload["metrics"]["sample_counts"]["evaluated"] else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate passive PAD on labeled videos"
    )
    parser.add_argument("--mode", choices=["calibrate", "evaluate"], default="evaluate")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--pad-model", required=True)
    parser.add_argument("--pad-model-version", required=True)
    parser.add_argument("--live-threshold", type=float)
    parser.add_argument("--threshold-version", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["calibration", "test"], default="test")
    parser.add_argument("--max-bpcer", type=float, default=0.05)
    parser.add_argument("--require-device-disjoint", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--min-valid-frames", type=int, default=5)
    parser.add_argument("--pad-input-size", type=int, default=224)
    parser.add_argument("--pad-live-class-index", type=int, default=1)
    parser.add_argument(
        "--pad-output-kind", choices=["logits", "probability"], default="logits"
    )
    parser.add_argument("--min-blur-variance", type=float, default=40.0)
    parser.add_argument("--min-brightness", type=float, default=35.0)
    parser.add_argument("--max-brightness", type=float, default=220.0)
    return parser


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


if __name__ == "__main__":
    raise SystemExit(main())
