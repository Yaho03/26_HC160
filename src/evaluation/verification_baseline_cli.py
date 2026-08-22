"""Generate a frozen threshold artifact and EXP-VER-001 clean report."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.reproducibility import stable_json_bytes
from src.evaluation.verification_calibration import (
    CalibrationError,
    VerificationScore,
    calibrate_threshold,
    clean_baseline_report_dict,
    evaluate_clean_baseline,
    threshold_artifact_dict,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-scores", type=Path, required=True)
    parser.add_argument("--test-scores", type=Path, required=True)
    parser.add_argument("--selection-method", choices=("target_far", "eer"), required=True)
    parser.add_argument("--target-far", type=float)
    parser.add_argument("--calibration-manifest-id", required=True)
    parser.add_argument("--calibration-manifest-sha256", required=True)
    parser.add_argument("--test-manifest-id", required=True)
    parser.add_argument("--test-manifest-sha256", required=True)
    parser.add_argument("--threshold-artifact-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--threshold-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--created-at")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_output_paths(
            (args.threshold_output, args.report_output),
            overwrite=args.overwrite,
        )
        calibration_scores = _load_scores(args.calibration_scores)
        test_scores = _load_scores(args.test_scores)
        calibration = calibrate_threshold(
            calibration_scores,
            method=args.selection_method,
            target_far=args.target_far,
        )
        result = evaluate_clean_baseline(
            calibration,
            test_scores,
            calibration_pair_ids=(record.pair_id for record in calibration_scores),
        )
        created_at = args.created_at or datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        threshold_artifact = threshold_artifact_dict(
            calibration,
            threshold_artifact_id=args.threshold_artifact_id,
            calibration_manifest_id=args.calibration_manifest_id,
            calibration_manifest_sha256=args.calibration_manifest_sha256,
            created_by_run_id=args.run_id,
            created_at=created_at,
        )
        report = clean_baseline_report_dict(
            result,
            calibration,
            threshold_artifact_id=args.threshold_artifact_id,
            test_manifest_id=args.test_manifest_id,
            test_manifest_sha256=args.test_manifest_sha256,
            generated_by_run_id=args.run_id,
            created_at=created_at,
        )
        _atomic_json_write(args.threshold_output, threshold_artifact, args.overwrite)
        _atomic_json_write(args.report_output, report, args.overwrite)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "threshold_artifact_id": args.threshold_artifact_id,
                    "threshold": calibration.threshold,
                    "test_pair_count": result.test_pair_count,
                    "far": result.rates.far.value,
                    "tar": result.rates.tar.value,
                },
                sort_keys=True,
            )
        )
        return 0
    except (CalibrationError, OSError, ValueError) as exc:
        parser.error(str(exc))
        return 2


def _load_scores(path: Path) -> tuple[VerificationScore, ...]:
    required = set(VerificationScore.__dataclass_fields__)
    records: list[VerificationScore] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CalibrationError(
                    f"invalid score JSON on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict) or set(value) != required:
                raise CalibrationError(
                    f"score line {line_number} must contain exactly: {', '.join(sorted(required))}"
                )
            records.append(VerificationScore(**value))
    return tuple(records)


def _validate_output_paths(paths: tuple[Path, ...], *, overwrite: bool) -> None:
    if len({path.resolve() for path in paths}) != len(paths):
        raise CalibrationError("threshold and report outputs must be different files")
    if not overwrite:
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise CalibrationError(
                f"refusing to overwrite existing file(s): {', '.join(existing)}"
            )


def _atomic_json_write(path: Path, value: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise CalibrationError(f"refusing to overwrite existing file: {path}")
    payload = stable_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
