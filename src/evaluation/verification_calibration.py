"""Threshold calibration and clean face-verification evaluation.

This module is dependency-free so protocol leakage and metric semantics can be
tested without downloading a face model. Scores are assumed to be similarities:
higher values indicate a stronger match and acceptance uses ``score >= threshold``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import inf, isfinite, nextafter
from typing import Any, Iterable

from src.common.reproducibility import stable_json_sha256
from src.evaluation.verification_metrics import (
    VerificationRates,
    verification_rates,
    wilson_interval,
)


class CalibrationError(ValueError):
    """Raised when score records violate the verification protocol."""


class UnsupportedFarTarget(CalibrationError):
    """Raised when too few impostor pairs support a requested FAR."""


@dataclass(frozen=True)
class VerificationScore:
    pair_id: str
    same_identity: bool
    score: float
    split: str
    protocol_id: str
    model_artifact_id: str
    preprocessing_artifact_id: str


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    rates: VerificationRates


@dataclass(frozen=True)
class EerEstimate:
    value: float
    threshold: float
    far: float
    frr: float


@dataclass(frozen=True)
class ThresholdCalibration:
    threshold: float
    selection_method: str
    target_far: float | None
    protocol_id: str
    model_artifact_id: str
    preprocessing_artifact_id: str
    calibration_pair_count: int
    genuine_pair_count: int
    impostor_pair_count: int
    calibration_pair_ids_sha256: str
    rates: VerificationRates
    roc_auc: float
    eer: EerEstimate


@dataclass(frozen=True)
class ConfidenceInterval:
    lower: float
    upper: float


@dataclass(frozen=True)
class VerificationConfidenceIntervals:
    far: ConfidenceInterval
    frr: ConfidenceInterval
    tar: ConfidenceInterval
    tnr: ConfidenceInterval
    accuracy: ConfidenceInterval


@dataclass(frozen=True)
class CleanBaselineResult:
    threshold: float
    test_pair_count: int
    genuine_pair_count: int
    impostor_pair_count: int
    test_pair_ids_sha256: str
    rates: VerificationRates
    confidence_intervals: VerificationConfidenceIntervals
    roc_auc: float
    descriptive_test_eer: EerEstimate


def calibrate_threshold(
    records: Iterable[VerificationScore],
    *,
    method: str,
    target_far: float | None = None,
) -> ThresholdCalibration:
    scores = _validate_records(records, required_split="calibration")
    protocol = _protocol_identity(scores)
    genuine_count, impostor_count = _label_counts(scores)
    points = operating_points(scores)
    auc = roc_auc(scores)
    eer = estimate_eer(points)

    if method == "target_far":
        if target_far is None:
            raise CalibrationError("target_far method requires target_far")
        _validate_target_far(target_far, impostor_count)
        selected = max(
            (point for point in points if _defined(point.rates.far.value) <= target_far),
            key=lambda point: (
                _defined(point.rates.tar.value),
                _defined(point.rates.far.value),
                -point.threshold,
            ),
        )
    elif method == "eer":
        if target_far is not None:
            raise CalibrationError("eer method must not define target_far")
        selected = min(
            points,
            key=lambda point: (
                abs(_defined(point.rates.far.value) - _defined(point.rates.frr.value)),
                (_defined(point.rates.far.value) + _defined(point.rates.frr.value)) / 2,
                -point.threshold,
            ),
        )
    else:
        raise CalibrationError(f"unsupported threshold selection method: {method}")

    return ThresholdCalibration(
        threshold=selected.threshold,
        selection_method=method,
        target_far=target_far,
        protocol_id=protocol[0],
        model_artifact_id=protocol[1],
        preprocessing_artifact_id=protocol[2],
        calibration_pair_count=len(scores),
        genuine_pair_count=genuine_count,
        impostor_pair_count=impostor_count,
        calibration_pair_ids_sha256=_pair_ids_hash(scores),
        rates=selected.rates,
        roc_auc=auc,
        eer=eer,
    )


def evaluate_clean_baseline(
    calibration: ThresholdCalibration,
    test_records: Iterable[VerificationScore],
    *,
    calibration_pair_ids: Iterable[str],
) -> CleanBaselineResult:
    scores = _validate_records(test_records, required_split="test")
    if _protocol_identity(scores) != (
        calibration.protocol_id,
        calibration.model_artifact_id,
        calibration.preprocessing_artifact_id,
    ):
        raise CalibrationError("test protocol/model/preprocessing differs from calibration")

    calibration_ids = tuple(calibration_pair_ids)
    if stable_json_sha256(sorted(calibration_ids)) != calibration.calibration_pair_ids_sha256:
        raise CalibrationError("calibration pair IDs do not match threshold provenance")
    overlap = sorted(set(calibration_ids) & {record.pair_id for record in scores})
    if overlap:
        raise CalibrationError(f"calibration/test pair overlap: {', '.join(overlap[:5])}")

    genuine_count, impostor_count = _label_counts(scores)
    decisions = tuple(record.score >= calibration.threshold for record in scores)
    rates = verification_rates(
        (record.same_identity for record in scores),
        decisions,
    )
    return CleanBaselineResult(
        threshold=calibration.threshold,
        test_pair_count=len(scores),
        genuine_pair_count=genuine_count,
        impostor_pair_count=impostor_count,
        test_pair_ids_sha256=_pair_ids_hash(scores),
        rates=rates,
        confidence_intervals=_confidence_intervals(rates),
        roc_auc=roc_auc(scores),
        descriptive_test_eer=estimate_eer(operating_points(scores)),
    )


def operating_points(records: Iterable[VerificationScore]) -> tuple[OperatingPoint, ...]:
    scores = tuple(records)
    if not scores:
        raise CalibrationError("at least one score record is required")
    thresholds = [nextafter(max(record.score for record in scores), inf)]
    thresholds.extend(sorted({record.score for record in scores}, reverse=True))
    labels = tuple(record.same_identity for record in scores)
    return tuple(
        OperatingPoint(
            threshold=threshold,
            rates=verification_rates(
                labels,
                (record.score >= threshold for record in scores),
            ),
        )
        for threshold in thresholds
    )


def estimate_eer(points: Iterable[OperatingPoint]) -> EerEstimate:
    materialized = tuple(points)
    if not materialized:
        raise CalibrationError("at least one operating point is required")
    selected = min(
        materialized,
        key=lambda point: (
            abs(_defined(point.rates.far.value) - _defined(point.rates.frr.value)),
            (_defined(point.rates.far.value) + _defined(point.rates.frr.value)) / 2,
            -point.threshold,
        ),
    )
    far = _defined(selected.rates.far.value)
    frr = _defined(selected.rates.frr.value)
    return EerEstimate(
        value=(far + frr) / 2,
        threshold=selected.threshold,
        far=far,
        frr=frr,
    )


def roc_auc(records: Iterable[VerificationScore]) -> float:
    """Return tie-aware ROC-AUC using average ranks."""
    scores = tuple(records)
    genuine_count, impostor_count = _label_counts(scores)
    ordered = sorted(scores, key=lambda record: record.score)
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end].score == ordered[index].score:
            end += 1
        average_rank = ((index + 1) + end) / 2
        positive_rank_sum += average_rank * sum(
            record.same_identity for record in ordered[index:end]
        )
        index = end
    return (
        positive_rank_sum - (genuine_count * (genuine_count + 1) / 2)
    ) / (genuine_count * impostor_count)


def far_target_supported(target_far: float, impostor_pair_count: int) -> bool:
    if not isinstance(target_far, (int, float)) or isinstance(target_far, bool):
        return False
    if not isfinite(target_far) or target_far <= 0 or target_far > 1:
        return False
    if type(impostor_pair_count) is not int or impostor_pair_count <= 0:
        return False
    return target_far * impostor_pair_count >= 1


def threshold_artifact_dict(
    calibration: ThresholdCalibration,
    *,
    threshold_artifact_id: str,
    calibration_manifest_id: str,
    calibration_manifest_sha256: str,
    model_artifact_sha256: str,
    preprocessing_artifact_sha256: str,
    calibration_score_export_sha256: str,
    created_by_run_id: str,
    created_at: str,
) -> dict[str, Any]:
    if not threshold_artifact_id or not calibration_manifest_id or not created_by_run_id:
        raise CalibrationError("threshold, calibration manifest, and run IDs are required")
    if len(calibration_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in calibration_manifest_sha256
    ):
        raise CalibrationError("calibration_manifest_sha256 must be lowercase SHA-256")
    _require_sha256("model_artifact_sha256", model_artifact_sha256)
    _require_sha256("preprocessing_artifact_sha256", preprocessing_artifact_sha256)
    _require_sha256(
        "calibration_score_export_sha256", calibration_score_export_sha256
    )
    return {
        "schema_version": "1.0",
        "threshold_artifact_id": threshold_artifact_id,
        "protocol_id": calibration.protocol_id,
        "model_artifact_id": calibration.model_artifact_id,
        "model_artifact_sha256": model_artifact_sha256,
        "preprocessing_artifact_id": calibration.preprocessing_artifact_id,
        "preprocessing_artifact_sha256": preprocessing_artifact_sha256,
        "score_function": "similarity",
        "score_direction": "higher_is_match",
        "decision_rule": "score_gte_threshold",
        "threshold": calibration.threshold,
        "selection_method": calibration.selection_method,
        "target_far": calibration.target_far,
        "calibration_manifest_id": calibration_manifest_id,
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "calibration_score_export_sha256": calibration_score_export_sha256,
        "calibration_pair_ids_sha256": calibration.calibration_pair_ids_sha256,
        "calibration_pair_count": calibration.calibration_pair_count,
        "genuine_pair_count": calibration.genuine_pair_count,
        "impostor_pair_count": calibration.impostor_pair_count,
        "achieved_far": asdict(calibration.rates.far),
        "achieved_tar": asdict(calibration.rates.tar),
        "calibration_roc_auc": calibration.roc_auc,
        "calibration_eer": asdict(calibration.eer),
        "created_by_run_id": created_by_run_id,
        "created_at": created_at,
    }


def clean_baseline_report_dict(
    result: CleanBaselineResult,
    calibration: ThresholdCalibration,
    *,
    threshold_artifact_id: str,
    test_manifest_id: str,
    test_manifest_sha256: str,
    model_artifact_sha256: str,
    preprocessing_artifact_sha256: str,
    test_score_export_sha256: str,
    generated_by_run_id: str,
    created_at: str,
) -> dict[str, Any]:
    if not threshold_artifact_id or not test_manifest_id or not generated_by_run_id:
        raise CalibrationError("threshold, test manifest, and run IDs are required")
    if len(test_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in test_manifest_sha256
    ):
        raise CalibrationError("test_manifest_sha256 must be lowercase SHA-256")
    _require_sha256("model_artifact_sha256", model_artifact_sha256)
    _require_sha256("preprocessing_artifact_sha256", preprocessing_artifact_sha256)
    _require_sha256("test_score_export_sha256", test_score_export_sha256)
    return {
        "schema_version": "1.0",
        "experiment_id": "EXP-VER-001",
        "report_id": f"clean_{result.test_pair_ids_sha256[:16]}",
        "protocol_id": calibration.protocol_id,
        "model_artifact_id": calibration.model_artifact_id,
        "model_artifact_sha256": model_artifact_sha256,
        "preprocessing_artifact_id": calibration.preprocessing_artifact_id,
        "preprocessing_artifact_sha256": preprocessing_artifact_sha256,
        "threshold_artifact_id": threshold_artifact_id,
        "frozen_threshold": result.threshold,
        "selection_method": calibration.selection_method,
        "calibration_target_far": calibration.target_far,
        "test_manifest_id": test_manifest_id,
        "test_manifest_sha256": test_manifest_sha256,
        "test_score_export_sha256": test_score_export_sha256,
        "test_pair_ids_sha256": result.test_pair_ids_sha256,
        "test_pair_count": result.test_pair_count,
        "genuine_pair_count": result.genuine_pair_count,
        "impostor_pair_count": result.impostor_pair_count,
        "confusion_counts": asdict(result.rates.counts),
        "rates": {
            name: asdict(getattr(result.rates, name))
            for name in ("far", "frr", "tar", "tnr", "accuracy")
        },
        "wilson_95_intervals": {
            name: asdict(getattr(result.confidence_intervals, name))
            for name in ("far", "frr", "tar", "tnr", "accuracy")
        },
        "test_roc_auc": result.roc_auc,
        "descriptive_test_eer": asdict(result.descriptive_test_eer),
        "generated_by_run_id": generated_by_run_id,
        "created_at": created_at,
        "limitations": [
            "The test EER is descriptive and did not alter the frozen threshold.",
            "This research result is not production financial authentication evidence.",
        ],
    }


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise CalibrationError(f"{name} must be lowercase SHA-256")


def _validate_records(
    records: Iterable[VerificationScore],
    *,
    required_split: str,
) -> tuple[VerificationScore, ...]:
    scores = tuple(records)
    if not scores:
        raise CalibrationError("at least one score record is required")
    pair_ids: set[str] = set()
    for record in scores:
        if not record.pair_id.strip():
            raise CalibrationError("pair_id must be non-empty")
        if record.pair_id in pair_ids:
            raise CalibrationError(f"duplicate pair_id: {record.pair_id}")
        pair_ids.add(record.pair_id)
        if type(record.same_identity) is not bool:
            raise CalibrationError("same_identity must be boolean")
        if not isinstance(record.score, (int, float)) or isinstance(record.score, bool):
            raise CalibrationError("score must be numeric")
        if not isfinite(record.score):
            raise CalibrationError("score must be finite")
        if record.score < -1 or record.score > 1:
            raise CalibrationError("cosine similarity score must be between -1 and 1")
        if record.split != required_split:
            raise CalibrationError(
                f"expected only {required_split} rows, received split={record.split}"
            )
        for field_name in (
            "protocol_id",
            "model_artifact_id",
            "preprocessing_artifact_id",
        ):
            if not getattr(record, field_name).strip():
                raise CalibrationError(f"{field_name} must be non-empty")
    _protocol_identity(scores)
    _label_counts(scores)
    return scores


def _protocol_identity(
    records: Iterable[VerificationScore],
) -> tuple[str, str, str]:
    identities = {
        (
            record.protocol_id,
            record.model_artifact_id,
            record.preprocessing_artifact_id,
        )
        for record in records
    }
    if len(identities) != 1:
        raise CalibrationError("score rows mix protocol, model, or preprocessing IDs")
    return next(iter(identities))


def _label_counts(records: Iterable[VerificationScore]) -> tuple[int, int]:
    materialized = tuple(records)
    genuine_count = sum(record.same_identity for record in materialized)
    impostor_count = len(materialized) - genuine_count
    if genuine_count == 0 or impostor_count == 0:
        raise CalibrationError("both genuine and impostor pairs are required")
    return genuine_count, impostor_count


def _pair_ids_hash(records: Iterable[VerificationScore]) -> str:
    return stable_json_sha256(sorted(record.pair_id for record in records))


def _validate_target_far(target_far: float, impostor_count: int) -> None:
    if not far_target_supported(target_far, impostor_count):
        minimum = 1 / impostor_count
        raise UnsupportedFarTarget(
            f"target FAR {target_far:g} is unsupported by {impostor_count} impostor pairs; "
            f"minimum empirical non-zero FAR is {minimum:g}"
        )


def _defined(value: float | None) -> float:
    if value is None:
        raise CalibrationError("rate is undefined because a required label class is absent")
    return value


def _confidence_intervals(rates: VerificationRates) -> VerificationConfidenceIntervals:
    def interval(name: str) -> ConfidenceInterval:
        rate = getattr(rates, name)
        bounds = wilson_interval(rate.numerator, rate.denominator)
        if bounds is None:
            raise CalibrationError(f"{name} confidence interval is undefined")
        return ConfidenceInterval(lower=bounds[0], upper=bounds[1])

    return VerificationConfidenceIntervals(
        far=interval("far"),
        frr=interval("frr"),
        tar=interval("tar"),
        tnr=interval("tnr"),
        accuracy=interval("accuracy"),
    )
