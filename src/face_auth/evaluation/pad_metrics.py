from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

from src.face_auth.domain import reason_codes
from src.face_auth.evaluation.calibration import CalibrationResult, calibrate_threshold


@dataclass(frozen=True)
class RateMetric:
    numerator: int
    denominator: int
    value: float | None
    ci95_low: float | None
    ci95_high: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def pad_metrics(sample_results: list[dict]) -> dict:
    evaluated = [
        result for result in sample_results if result["outcome"] in {"PASS", "FAIL"}
    ]
    bona_fide = [result for result in evaluated if result["label"] == "bona_fide"]
    attacks = [result for result in evaluated if result["label"] == "attack"]
    bpcer = _rate(
        sum(result["outcome"] == "FAIL" for result in bona_fide), len(bona_fide)
    )
    apcer = _rate(sum(result["outcome"] == "PASS" for result in attacks), len(attacks))
    acer = (
        None
        if bpcer.value is None or apcer.value is None
        else (bpcer.value + apcer.value) / 2.0
    )

    species = {}
    all_attacks = [result for result in sample_results if result["label"] == "attack"]
    for name in sorted({result["attack_species"] for result in all_attacks}):
        attempted = [
            result for result in all_attacks if result["attack_species"] == name
        ]
        subset = [
            result for result in attempted if result["outcome"] in {"PASS", "FAIL"}
        ]
        species_rate = _rate(
            sum(result["outcome"] == "PASS" for result in subset), len(subset)
        ).to_dict()
        species_rate.update(
            {
                "presentations": len(attempted),
                "not_evaluated": sum(
                    result["outcome"] == "NOT_EVALUATED" for result in attempted
                ),
                "error": sum(result["outcome"] == "ERROR" for result in attempted),
            }
        )
        species[name] = species_rate

    defined_species = [
        (name, metric["value"])
        for name, metric in species.items()
        if metric["value"] is not None
    ]
    worst_species = (
        None
        if not defined_species
        else {
            "attack_species": max(defined_species, key=lambda item: item[1])[0],
            "value": max(value for _, value in defined_species),
        }
    )
    worst_acer = (
        None
        if worst_species is None or bpcer.value is None
        else (worst_species["value"] + bpcer.value) / 2.0
    )

    return {
        "apcer": apcer.to_dict(),
        "bpcer": bpcer.to_dict(),
        "acer": acer,
        "apcer_by_attack_species": species,
        "apcer_worst_species": worst_species,
        "acer_worst_species": worst_acer,
        "sample_counts": {
            "total": len(sample_results),
            "evaluated": len(evaluated),
            "not_evaluated": sum(
                result["outcome"] == "NOT_EVALUATED" for result in sample_results
            ),
            "error": sum(result["outcome"] == "ERROR" for result in sample_results),
            "bona_fide_evaluated": len(bona_fide),
            "attack_evaluated": len(attacks),
        },
    }


def calibrate_pad_threshold(
    sample_results: list[dict], *, max_bpcer: float
) -> CalibrationResult:
    scored = [
        result
        for result in sample_results
        if result["outcome"] in {"PASS", "FAIL"} and result["score"] is not None
    ]
    return calibrate_threshold(
        [result["score"] for result in scored if result["label"] == "bona_fide"],
        [result["score"] for result in scored if result["label"] == "attack"],
        higher_is_clean=True,
        max_clean_frr=max_bpcer,
    )


def reclassify_pad_samples(sample_results: list[dict], threshold: float) -> list[dict]:
    reclassified = []
    for result in sample_results:
        updated = dict(result)
        updated["threshold"] = threshold
        if result["outcome"] in {"PASS", "FAIL"} and result["score"] is not None:
            passed = result["score"] >= threshold
            updated["outcome"] = "PASS" if passed else "FAIL"
            updated["reason_codes"] = [] if passed else [reason_codes.SPOOF_SUSPECTED]
        reclassified.append(updated)
    return reclassified


def _rate(numerator: int, denominator: int) -> RateMetric:
    if denominator == 0:
        return RateMetric(numerator, denominator, None, None, None)
    value = numerator / denominator
    low, high = _wilson_interval(numerator, denominator)
    return RateMetric(numerator, denominator, value, low, high)


def _wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)
