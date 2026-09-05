"""
방어 전후 인증 결과 비교

09_EVALUATION_METRICS.md 3절의 세 지표는 분모가 다르다. 같은 label을 쓰면 안 되므로
이름과 분모를 함께 반환한다.

    conditional defense success rate = 방어 후 reject / 방어 전 accept
    conditional ASR after defense    = 방어 후 accept / 방어 전 accept
    population ASR after defense     = 방어 후 accept / 전체 eligible attempt

07_DEFENSE_AND_DETECTION_SPEC.md 7절의 잠정 통과 기준은 conditional ASR 50% 이상
감소와 고정 threshold에서 clean TAR 감소 2%p 이하다.
"""

from __future__ import annotations

import numpy as np

# 07_DEFENSE_AND_DETECTION_SPEC.md 7절
ASR_REDUCTION_BUDGET = 0.50
CLEAN_COST_BUDGET_PP = 2.0


class NoEligibleAttackError(ValueError):
    """방어 전에 accept된 공격이 없다. conditional 지표의 분모가 0이다."""


def conditional_defense_metrics(
    *,
    attack_similarity,
    attack_detected,
    identity_threshold: float,
    attack_model: str = "unspecified",
) -> dict:
    """
    Eligible attempt는 방어 전 accept된 공격이다(09 2절). 방어 전에 이미 거부된
    공격을 분모에 넣으면 방어 성능이 부풀려진다.

    detector가 hit이면 optional veto로 SECURITY_DENIED가 되므로 reject로 본다.

    attack_model은 이 판정이 어떤 공격에서 성립하는지를 남긴다. 같은 방어가
    비적응 공격에서 conditional ASR 감소 67%, BPDA에서 0%를 기록했다. 공격 모델
    없이 판정만 기록하면 조건 없는 주장이 된다. 기본값은 unspecified이며
    비적응이라고 가정하지 않는다.
    """
    similarity = np.asarray(attack_similarity, dtype=float)
    detected = np.asarray(attack_detected, dtype=bool)
    if similarity.shape != detected.shape:
        raise ValueError("attack_similarity와 attack_detected의 길이가 다르다")

    accepted_before = similarity >= identity_threshold
    eligible = int(accepted_before.sum())
    if eligible == 0:
        raise NoEligibleAttackError(
            "방어 전 accept된 공격이 없다. conditional 지표를 계산할 수 없다. "
            "공격 예산이나 신원 임계값을 확인하라."
        )

    rejected_after = int((accepted_before & detected).sum())
    accepted_after = eligible - rejected_after

    # 방어 전 conditional ASR은 정의상 1.0이다. eligible 자체가 성공한 공격이므로.
    before = 1.0
    after = accepted_after / eligible
    reduction = before - after

    return {
        "attack_model": attack_model,
        "identity_threshold": identity_threshold,
        "total_attempts": int(similarity.size),
        "eligible": eligible,
        "rejected_after_defense": rejected_after,
        "accepted_after_defense": accepted_after,
        "conditional_asr_before_defense": before,
        "conditional_asr_after_defense": after,
        "conditional_defense_success_rate": rejected_after / eligible,
        "population_asr_after_defense": accepted_after / similarity.size,
        "conditional_asr_reduction": reduction,
        "meets_asr_budget": bool(reduction >= ASR_REDUCTION_BUDGET),
    }


def clean_cost(
    *,
    clean_similarity,
    clean_detected,
    identity_threshold: float,
) -> dict:
    """
    방어가 정상 사용자에게 물리는 비용. 방어 전에 이미 거부된 clean은 방어 탓이
    아니므로 분자에서 뺀다.
    """
    similarity = np.asarray(clean_similarity, dtype=float)
    detected = np.asarray(clean_detected, dtype=bool)
    if similarity.shape != detected.shape:
        raise ValueError("clean_similarity와 clean_detected의 길이가 다르다")

    total = int(similarity.size)
    if total == 0:
        raise ValueError("clean 표본이 없다")

    accepted_before_mask = similarity >= identity_threshold
    accepted_before = int(accepted_before_mask.sum())
    accepted_after = int((accepted_before_mask & ~detected).sum())

    tar_before = accepted_before / total
    tar_after = accepted_after / total
    delta_pp = (tar_after - tar_before) * 100.0

    return {
        "identity_threshold": identity_threshold,
        "n_clean": total,
        "accepted_before": accepted_before,
        "accepted_after": accepted_after,
        "clean_tar_before": tar_before,
        "clean_tar_after": tar_after,
        "clean_tar_delta_pp": delta_pp,
        "meets_clean_budget": bool(abs(delta_pp) <= CLEAN_COST_BUDGET_PP),
    }
