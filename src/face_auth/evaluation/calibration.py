from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float
    clean_false_reject_rate: float
    attack_false_accept_rate: float
    clean_count: int
    attack_count: int
    direction: str

    def to_dict(self) -> dict:
        return asdict(self)


def calibrate_threshold(
    clean_scores: Iterable[float],
    attack_scores: Iterable[float],
    *,
    higher_is_clean: bool,
    max_clean_frr: float = 0.05,
) -> CalibrationResult:
    """Selects a threshold from validation scores under a clean-FRR budget."""
    clean = sorted(float(score) for score in clean_scores)
    attack = [float(score) for score in attack_scores]
    if not clean or not attack:
        raise ValueError("Both clean and attack validation scores are required")
    if not 0.0 <= max_clean_frr < 1.0:
        raise ValueError("max_clean_frr must be in [0, 1)")

    if higher_is_clean:
        index = min(len(clean) - 1, int(max_clean_frr * len(clean)))
        threshold = clean[index]
        clean_rejected = sum(score < threshold for score in clean)
        attacks_accepted = sum(score >= threshold for score in attack)
        direction = "PASS_IF_GREATER_OR_EQUAL"
    else:
        descending = sorted(clean, reverse=True)
        index = min(len(descending) - 1, int(max_clean_frr * len(descending)))
        threshold = descending[index]
        clean_rejected = sum(score > threshold for score in clean)
        attacks_accepted = sum(score <= threshold for score in attack)
        direction = "PASS_IF_LESS_OR_EQUAL"

    return CalibrationResult(
        threshold=threshold,
        clean_false_reject_rate=clean_rejected / len(clean),
        attack_false_accept_rate=attacks_accepted / len(attack),
        clean_count=len(clean),
        attack_count=len(attack),
        direction=direction,
    )
