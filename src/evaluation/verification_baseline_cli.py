"""Generate a frozen threshold artifact and EXP-VER-001 clean report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
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
    parser.add_argument("--calibration-score-metadata", type=Path, required=True)
    parser.add_argument("--test-scores", type=Path, required=True)
    parser.add_argument("--test-score-metadata", type=Path, required=True)
    parser.add_argument("--selection-method", choices=("target_far", "eer"), required=True)
    parser.add_argument("--target-far", type=float)
    parser.add_argument("--threshold-artifact-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--threshold-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--created-at")
    parser.add_argument("--overwrite", action="store_true")
    return parser


@dataclass(frozen=True)
class LoadedScoreFile:
    records: tuple[VerificationScore, ...]
    sha256: str
    bytes: int


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_output_paths(
            (args.threshold_output, args.report_output),
            overwrite=args.overwrite,
        )
        calibration_file = _load_scores(args.calibration_scores)
        test_file = _load_scores(args.test_scores)
        calibration_scores = calibration_file.records
        test_scores = test_file.records
        calibration_metadata, calibration_metadata_sha256 = _load_score_export_metadata(
            args.calibration_score_metadata,
            loaded_score_file=calibration_file,
            expected_split="calibration",
        )
        test_metadata, test_metadata_sha256 = _load_score_export_metadata(
            args.test_score_metadata,
            loaded_score_file=test_file,
            expected_split="test",
        )
        _validate_matching_exports(calibration_metadata, test_metadata)
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
            calibration_manifest_id=calibration_metadata["pair_manifest"]["id"],
            calibration_manifest_sha256=calibration_metadata["pair_manifest"][
                "sha256"
            ],
            model_artifact_sha256=calibration_metadata["model_artifact"]["sha256"],
            preprocessing_artifact_sha256=calibration_metadata[
                "preprocessing_artifact"
            ]["sha256"],
            calibration_score_export_sha256=calibration_metadata_sha256,
            created_by_run_id=args.run_id,
            created_at=created_at,
        )
        report = clean_baseline_report_dict(
            result,
            calibration,
            threshold_artifact_id=args.threshold_artifact_id,
            test_manifest_id=test_metadata["pair_manifest"]["id"],
            test_manifest_sha256=test_metadata["pair_manifest"]["sha256"],
            model_artifact_sha256=test_metadata["model_artifact"]["sha256"],
            preprocessing_artifact_sha256=test_metadata["preprocessing_artifact"][
                "sha256"
            ],
            test_score_export_sha256=test_metadata_sha256,
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


def _load_scores(path: Path) -> LoadedScoreFile:
    required = set(VerificationScore.__dataclass_fields__)
    records: list[VerificationScore] = []
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CalibrationError("score file must be UTF-8 JSONL") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
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
    return LoadedScoreFile(
        records=tuple(records),
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
    )


def _load_score_export_metadata(
    path: Path,
    *,
    loaded_score_file: LoadedScoreFile,
    expected_split: str,
) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"invalid score metadata JSON: {exc.msg}") from exc
    required = {
        "schema_version",
        "export_id",
        "run_id",
        "created_at",
        "code",
        "protocol_id",
        "split",
        "software",
        "model_artifact",
        "preprocessing_artifact",
        "dataset_manifest",
        "pair_manifest",
        "score_file",
        "evaluation_config",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise CalibrationError("score metadata fields do not match the export schema")
    if value["schema_version"] != "1.0" or value["split"] != expected_split:
        raise CalibrationError(f"score metadata must describe split={expected_split}")
    for field in (
        "code",
        "software",
        "model_artifact",
        "preprocessing_artifact",
        "dataset_manifest",
        "pair_manifest",
        "score_file",
        "evaluation_config",
    ):
        if not isinstance(value[field], dict):
            raise CalibrationError(f"{field} metadata must be an object")
    nested_fields = {
        "code": {"git_commit", "dirty_worktree"},
        "software": {"python", "torch", "facenet_pytorch", "numpy", "pillow"},
        "model_artifact": {
            "id",
            "sha256",
            "bytes",
            "architecture",
            "weights_source",
        },
        "preprocessing_artifact": {"id", "sha256", "bytes"},
        "dataset_manifest": {"id", "sha256", "row_count"},
        "pair_manifest": {"id", "sha256", "row_count", "split"},
        "score_file": {"sha256", "bytes", "row_count"},
        "evaluation_config": {
            "device",
            "batch_size",
            "require_identity_disjoint",
            "deterministic_algorithms",
        },
    }
    for field, fields in nested_fields.items():
        if set(value[field]) != fields:
            raise CalibrationError(f"{field} fields do not match the export schema")
    if value["code"]["dirty_worktree"] is not False:
        raise CalibrationError("reportable score exports require a clean worktree")
    if value["evaluation_config"]["require_identity_disjoint"] is not True:
        raise CalibrationError("reportable score exports require identity-disjoint data")
    if value["evaluation_config"]["deterministic_algorithms"] is not True:
        raise CalibrationError("reportable score exports require deterministic inference")
    score_metadata = value["score_file"]
    if score_metadata.get("sha256") != loaded_score_file.sha256:
        raise CalibrationError("score file hash differs from its export metadata")
    if score_metadata.get("bytes") != loaded_score_file.bytes:
        raise CalibrationError("score file byte count differs from its export metadata")
    if score_metadata.get("row_count") != len(loaded_score_file.records):
        raise CalibrationError("score row count differs from its export metadata")
    expected_protocol = {
        (
            record.protocol_id,
            record.model_artifact_id,
            record.preprocessing_artifact_id,
            record.split,
        )
        for record in loaded_score_file.records
    }
    metadata_protocol = (
        value["protocol_id"],
        value["model_artifact"].get("id"),
        value["preprocessing_artifact"].get("id"),
        value["split"],
    )
    if expected_protocol != {metadata_protocol}:
        raise CalibrationError("score rows differ from their export provenance")
    if value["pair_manifest"].get("row_count") != len(loaded_score_file.records):
        raise CalibrationError("pair manifest count differs from score rows")
    if value["pair_manifest"].get("split") != expected_split:
        raise CalibrationError("pair manifest split differs from score metadata")
    return value, hashlib.sha256(payload).hexdigest()


def _validate_matching_exports(
    calibration: dict[str, Any], test: dict[str, Any]
) -> None:
    for field in ("protocol_id",):
        if calibration[field] != test[field]:
            raise CalibrationError(f"calibration/test {field} differs")
    for field in ("code", "software", "evaluation_config"):
        if calibration[field] != test[field]:
            raise CalibrationError(f"calibration/test {field} differs")
    for field in ("model_artifact", "preprocessing_artifact"):
        left = calibration[field]
        right = test[field]
        if (left.get("id"), left.get("sha256")) != (
            right.get("id"),
            right.get("sha256"),
        ):
            raise CalibrationError(f"calibration/test {field} differs")
    left_dataset = calibration["dataset_manifest"]
    right_dataset = test["dataset_manifest"]
    if (left_dataset.get("id"), left_dataset.get("sha256")) != (
        right_dataset.get("id"),
        right_dataset.get("sha256"),
    ):
        raise CalibrationError("calibration/test dataset_manifest differs")


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
