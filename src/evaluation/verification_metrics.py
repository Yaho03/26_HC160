"""Metric definitions shared by new verification experiments.

Legacy report generators remain unchanged. New experiments should use this
module so numerators and denominators are explicit and consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RateEstimate:
    numerator: int
    denominator: int
    value: float | None


@dataclass(frozen=True)
class ConfusionCounts:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    @property
    def total(self) -> int:
        return self.true_positive + self.true_negative + self.false_positive + self.false_negative


@dataclass(frozen=True)
class VerificationRates:
    counts: ConfusionCounts
    far: RateEstimate
    frr: RateEstimate
    tar: RateEstimate
    tnr: RateEstimate
    accuracy: RateEstimate


@dataclass(frozen=True)
class AttackTransitionSummary:
    total_attempts: int
    eligible_reject_before: int
    reject_to_accept: int
    stayed_rejected: int
    already_accepted_before: int
    targeted_asr: RateEstimate


@dataclass(frozen=True)
class DefenseTransitionSummary:
    total_attack_rows: int
    successful_before_defense: int
    blocked_after_defense: int
    still_accepted_after_defense: int
    conditional_defense_success_rate: RateEstimate
    conditional_asr_after_defense: RateEstimate
    population_asr_after_defense: RateEstimate


def _as_bools(values: Iterable[bool], name: str) -> tuple[bool, ...]:
    result = tuple(values)
    if any(type(value) is not bool for value in result):
        raise TypeError(f"{name} must contain booleans only")
    return result


def _rate(numerator: int, denominator: int) -> RateEstimate:
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise ValueError(f"Invalid rate: {numerator}/{denominator}")
    return RateEstimate(
        numerator=numerator,
        denominator=denominator,
        value=None if denominator == 0 else numerator / denominator,
    )


def verification_rates(
    same_identity: Sequence[bool] | Iterable[bool],
    accepted: Sequence[bool] | Iterable[bool],
) -> VerificationRates:
    labels = _as_bools(same_identity, "same_identity")
    decisions = _as_bools(accepted, "accepted")
    if len(labels) != len(decisions):
        raise ValueError("same_identity and accepted must have equal length")

    tp = tn = fp = fn = 0
    for same, accept in zip(labels, decisions):
        if same and accept:
            tp += 1
        elif same:
            fn += 1
        elif accept:
            fp += 1
        else:
            tn += 1

    counts = ConfusionCounts(tp, tn, fp, fn)
    positives = tp + fn
    negatives = tn + fp
    return VerificationRates(
        counts=counts,
        far=_rate(fp, negatives),
        frr=_rate(fn, positives),
        tar=_rate(tp, positives),
        tnr=_rate(tn, negatives),
        accuracy=_rate(tp + tn, counts.total),
    )


def attack_transition_summary(
    accepted_before: Sequence[bool] | Iterable[bool],
    accepted_after: Sequence[bool] | Iterable[bool],
) -> AttackTransitionSummary:
    before = _as_bools(accepted_before, "accepted_before")
    after = _as_bools(accepted_after, "accepted_after")
    if len(before) != len(after):
        raise ValueError("accepted_before and accepted_after must have equal length")

    eligible = sum(not value for value in before)
    successes = sum((not left) and right for left, right in zip(before, after))
    already_accepted = sum(before)
    return AttackTransitionSummary(
        total_attempts=len(before),
        eligible_reject_before=eligible,
        reject_to_accept=successes,
        stayed_rejected=eligible - successes,
        already_accepted_before=already_accepted,
        targeted_asr=_rate(successes, eligible),
    )


def defense_transition_summary(
    accepted_before_defense: Sequence[bool] | Iterable[bool],
    accepted_after_defense: Sequence[bool] | Iterable[bool],
) -> DefenseTransitionSummary:
    before = _as_bools(accepted_before_defense, "accepted_before_defense")
    after = _as_bools(accepted_after_defense, "accepted_after_defense")
    if len(before) != len(after):
        raise ValueError("accepted_before_defense and accepted_after_defense must have equal length")

    successful_before = sum(before)
    blocked = sum(left and not right for left, right in zip(before, after))
    still_accepted = sum(left and right for left, right in zip(before, after))
    population_accepted_after = sum(after)

    return DefenseTransitionSummary(
        total_attack_rows=len(before),
        successful_before_defense=successful_before,
        blocked_after_defense=blocked,
        still_accepted_after_defense=still_accepted,
        conditional_defense_success_rate=_rate(blocked, successful_before),
        conditional_asr_after_defense=_rate(still_accepted, successful_before),
        population_asr_after_defense=_rate(population_accepted_after, len(before)),
    )


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """Return a Wilson score interval for a binomial proportion."""
    if successes < 0 or total < 0 or successes > total:
        raise ValueError(f"Invalid binomial counts: {successes}/{total}")
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + (z * z) / total
    center = (p + (z * z) / (2 * total)) / denominator
    margin = (
        z
        * sqrt((p * (1 - p) / total) + (z * z) / (4 * total * total))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)
