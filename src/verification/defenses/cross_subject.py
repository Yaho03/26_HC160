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


class NoEnrollmentSessionError(ValueError):
    """세션이 하나뿐이라 등록과 테스트를 나눌 수 없다."""


class MissingSessionTimeError(ValueError):
    """세션 촬영 시각이 없어 등록 세션을 정할 수 없다."""


NORMALIZATIONS = ("population", "per_user")

# 등록 배정. 피험자당 세션이 둘뿐이면 이 둘이 가능한 배정의 전부다.
ENROLLMENTS = ("earliest", "latest")


def session_times_from_sidecars(sidecars):
    """
    세션 사이드카에서 session_id → created_at 표를 만든다.

    probe.csv 행에는 촬영 시각이 없다. 등록 세션을 시각으로 고르려면 사이드카에서
    따로 받아 넘겨야 한다.
    """
    times = {}
    for meta in sidecars:
        session_id = meta.get("session_id")
        if session_id is None:
            continue
        created_at = meta.get("created_at")
        if created_at is None:
            raise MissingSessionTimeError(
                f"세션 {session_id}의 사이드카에 created_at이 없다."
            )
        times[session_id] = created_at
    return times


def order_sessions(sessions, session_times):
    """
    세션을 촬영 시각 오름차순으로 놓는다.

    세션 ID는 무작위 16진 문자열이라 정렬해도 촬영 순서가 나오지 않는다. 시각이
    없으면 ID 정렬로 되돌아가지 않고 거부한다. 조용한 대체가 의도와 구현이 어긋난
    채로 지나가게 만든 원인이었다.

    시각이 같으면 세션 ID로 가른다. 임의의 선택이지만 실행마다 바뀌지는 않는다.
    """
    sessions = list(sessions)
    missing = sorted(
        s for s in sessions if session_times.get(s) is None
    )
    if missing:
        raise MissingSessionTimeError(
            f"촬영 시각을 모르는 세션이 있다: {missing}. "
            "사이드카의 created_at을 session_times로 넘겨라."
        )
    return sorted(sessions, key=lambda s: (session_times[s], s))


def enrollment_split(rows, subject, *, session_times, enrollment="earliest"):
    """
    한 피험자의 행을 등록 세션과 테스트 세션으로 나눈다.

    같은 세션으로 정규화하고 평가하면 피험자 내부 누수가 된다. 실제 배포에서도
    등록은 인증 이전에 일어나므로 등록 세션은 촬영 시각으로 고른다.

    enrollment는 어느 쪽을 등록으로 볼지 정한다. 피험자당 세션이 둘뿐인 현재
    데이터에서 "첫 번째"는 2원소 집합에서의 선택이며 레버리지가 크다. 두 배정을
    모두 돌려 결과를 범위로 보고하기 위해 배정을 인자로 둔다.
    """
    if enrollment not in ENROLLMENTS:
        raise ValueError(
            f"알 수 없는 등록 배정 {enrollment!r}. 사용 가능: {list(ENROLLMENTS)}"
        )
    subject_rows = [row for row in rows if row["subject_id"] == subject]
    sessions = {row["session_id"] for row in subject_rows}
    if len(sessions) < 2:
        raise NoEnrollmentSessionError(
            f"피험자 {subject}의 세션이 {len(sessions)}개다. 등록과 테스트를 나누려면 "
            "두 개 이상이 필요하다."
        )
    ordered = order_sessions(sessions, session_times)
    chosen = ordered[0] if enrollment == "earliest" else ordered[-1]
    return (
        [row for row in subject_rows if row["session_id"] == chosen],
        [row for row in subject_rows if row["session_id"] != chosen],
    )


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


def leave_one_subject_out(
    rows, *, features, target_fpr=0.01, window_frames=3, normalization="population",
    session_times=None, enrollment="earliest",
):
    """
    피험자를 한 명씩 빼며 검증한다. 임계값은 항상 학습 피험자에서만 나온다.

    normalization이 population이면 학습 피험자의 clean 통계로 모두를 정규화한다.
    per_user면 각자 자기 등록 세션 통계로 정규화한다. 등록 통계는 배포 환경에서
    실제로 쓸 수 있는 정보이므로 누수가 아니며, 등록 세션은 테스트에서 제외한다.

    per_user는 등록 분할을 쓰므로 session_times가 있어야 한다. 없이 돌면 등록
    세션을 ID 정렬로 고르게 되고, 그것이 촬영 순서와 무관하다는 것이 이 인자를
    필수로 만든 이유다.

    분모가 0이면 rate를 None으로 둔다. 0으로 대체하면 표본 부족을 성능으로 오해한다.
    """
    if normalization not in NORMALIZATIONS:
        raise ValueError(
            f"알 수 없는 정규화 {normalization!r}. 사용 가능: {list(NORMALIZATIONS)}"
        )
    if normalization != "population" and session_times is None:
        raise MissingSessionTimeError(
            f"{normalization} 정규화는 등록 분할을 쓴다. 사이드카의 created_at을 "
            "session_times로 넘겨라."
        )

    subjects = {row["subject_id"] for row in rows}
    results = []

    for train_subjects, test_subject in subject_splits(subjects):
        if normalization == "population":
            train_rows = [r for r in rows if r["subject_id"] in train_subjects]
            test_rows = [r for r in rows if r["subject_id"] == test_subject]
            train_stats = _statistics(train_rows, features)
            train_scored = _scores(train_rows, "clean", features, train_stats, window_frames)
            clean = _scores(test_rows, "clean", features, train_stats, window_frames)
            adversarial = _scores(
                test_rows, "adversarial", features, train_stats, window_frames
            )
        else:
            # 각자 자기 등록 세션으로 정규화하고, 등록 세션은 평가에서 뺀다.
            train_scored_parts = []
            for subject in train_subjects:
                enrolled, test = enrollment_split(
                    rows, subject, session_times=session_times, enrollment=enrollment
                )
                stats = _statistics(enrolled, features)
                train_scored_parts.append(
                    _scores(test, "clean", features, stats, window_frames)
                )
            train_scored = np.concatenate(train_scored_parts)

            enrolled, test_rows = enrollment_split(
                rows, test_subject, session_times=session_times, enrollment=enrollment
            )
            stats = _statistics(enrolled, features)
            clean = _scores(test_rows, "clean", features, stats, window_frames)
            adversarial = _scores(
                test_rows, "adversarial", features, stats, window_frames
            )

        threshold = threshold_at_fpr(train_scored, target_fpr)

        results.append(
            {
                "normalization": normalization,
                "enrollment": enrollment if normalization != "population" else None,
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
