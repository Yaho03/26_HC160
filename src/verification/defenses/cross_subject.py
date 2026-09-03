"""
피험자 분리 검증

07_DEFENSE_AND_DETECTION_SPEC.md 5절은 subject와 session 분리를 요구한다. 같은
피험자로 임계값을 정하고 같은 피험자로 평가하면 낙관 편향이 생긴다. 실제 서비스에서
detector가 만나는 것은 임계값 산출에 쓰이지 않은 사용자다.

임계값과 정규화 통계 모두 학습 피험자의 clean 표본에서만 나온다. 테스트 피험자의
표본이 어느 쪽에도 들어가면 분리가 깨진다.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.verification.defenses.probe_analyze import roc_auc, threshold_at_fpr
from src.verification.defenses.probe_threshold import aggregate_by_session, sample_key


class InsufficientSubjectsError(ValueError):
    """피험자가 한 명이면 held-out 구성을 만들 수 없다."""


def subject_splits(subjects):
    """피험자마다 한 번씩 테스트로 빼고 나머지를 학습으로 쓴다."""
    subjects = sorted(set(subjects))
    if len(subjects) < 2:
        raise InsufficientSubjectsError(
            f"피험자가 {len(subjects)}명이다. 분리 검증에는 두 명 이상이 필요하다."
        )
    for held_out in subjects:
        yield [s for s in subjects if s != held_out], held_out


def _statistics(rows, features):
    """학습 피험자의 clean 표본에서만 정규화 통계를 만든다."""
    buckets = defaultdict(list)
    for row in rows:
        if row["label"] != "clean":
            continue
        for transform, measure in features:
            if row["transform"] == transform:
                buckets[(transform, measure)].append(row[measure])

    statistics = {}
    for key, values in buckets.items():
        array = np.asarray(values, dtype=float)
        deviation = array.std()
        statistics[key] = (array.mean(), deviation if deviation > 1e-12 else 1.0)
    return statistics


def _scores(rows, label, features, statistics, window_frames):
    """표본별 결합 점수를 세션 경계를 지켜 집계한다."""
    order, seen = [], set()
    values = defaultdict(dict)
    for row in rows:
        if row["label"] != label:
            continue
        key = sample_key(row)
        if key not in seen:
            seen.add(key)
            order.append((key, row["session_id"]))
        for transform, measure in features:
            if row["transform"] == transform:
                mean, deviation = statistics[(transform, measure)]
                values[key][(transform, measure)] = (row[measure] - mean) / deviation

    combined = np.asarray([sum(values[key].values()) for key, _ in order], dtype=float)
    sessions = [session for _, session in order]
    return aggregate_by_session(combined, sessions, window_frames)


def leave_one_subject_out(rows, *, features, target_fpr=0.01, window_frames=3):
    """
    피험자를 한 명씩 빼며 검증한다. 임계값과 통계는 학습 피험자에서만 나온다.

    분모가 0이면 rate를 None으로 둔다. 0으로 대체하면 표본 부족을 성능으로 오해한다.
    """
    subjects = {row["subject_id"] for row in rows}
    results = []

    for train_subjects, test_subject in subject_splits(subjects):
        train_rows = [r for r in rows if r["subject_id"] in train_subjects]
        test_rows = [r for r in rows if r["subject_id"] == test_subject]

        statistics = _statistics(train_rows, features)
        threshold = threshold_at_fpr(
            _scores(train_rows, "clean", features, statistics, window_frames), target_fpr
        )
        clean = _scores(test_rows, "clean", features, statistics, window_frames)
        adversarial = _scores(test_rows, "adversarial", features, statistics, window_frames)

        results.append(
            {
                "train_subjects": train_subjects,
                "test_subject": test_subject,
                "threshold": float(threshold) if threshold is not None else None,
                "n_clean": int(clean.size),
                "n_adversarial": int(adversarial.size),
                "fpr": float(np.mean(clean >= threshold)) if clean.size else None,
                "tpr": float(np.mean(adversarial >= threshold)) if adversarial.size else None,
                "roc_auc": roc_auc(adversarial, clean),
            }
        )
    return results
