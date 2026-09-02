"""
Detector threshold artifact 생성

계측 CSV와 세션 사이드카에서 detector operating threshold를 산출하고
schemas/detector-threshold-artifact.schema.json 형식으로 기록한다.

임계값은 clean 표본만으로 정한다. adversarial 표본은 evaluation 블록에만 들어간다.
구조 자체가 이 분리를 드러내도록 calibration과 evaluation을 나눴다.

표본이 목표 FPR을 만족할 수 없으면 임계값을 만들지 않고 예외를 낸다. 조용히 FPR 0으로
낮추면 근거 없는 임계값이 artifact로 굳는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.verification.defenses.conditional_asr import (
    ASR_REDUCTION_BUDGET,
    NoEligibleAttackError,
    clean_cost,
    conditional_defense_metrics,
)
from src.verification.defenses.probe_analyze import (
    MEASURES,
    combine_clean_normalized,
    detector_metrics,
    feature_table,
    load_probe_rows,
    roc_auc,
    threshold_at_fpr,
)

SCHEMA_VERSION = "detector-threshold-artifact/1.0"

# 07_DEFENSE_AND_DETECTION_SPEC.md 7절: 고정 threshold에서 clean TAR 감소 2%p 이하
CLEAN_COST_BUDGET_PP = 2.0


class InsufficientCleanSamplesError(ValueError):
    """clean 표본 수가 목표 FPR을 만족할 수 없다."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_limitations(
    sidecars,
    *,
    subjects,
    sessions,
    clean_tar_delta_pp: float | None = None,
    asr_reduction: float | None = None,
) -> list[str]:
    """검증되지 않은 조건을 자동으로 남긴다. 사람이 적기를 기다리지 않는다."""
    limitations = []
    if len(subjects) < 2:
        limitations.append(
            f"단일 피험자({len(subjects)}명)로 산출했다. 피험자 간 일반화 근거가 없다."
        )
    if len(sessions) < 2:
        limitations.append(
            f"단일 세션({len(sessions)}개)으로 산출했다. 조명과 기기 변화를 반영하지 못한다."
        )
    attack_kinds = {
        sidecar.get("attack", {}).get("kind")
        for sidecar in sidecars
        if sidecar.get("attack", {}).get("kind")
    }
    if len(attack_kinds) < 2:
        limitations.append(
            f"단일 공격 종류({', '.join(sorted(attack_kinds)) or '미상'})만 평가했다. "
            "다른 공격으로의 일반화 근거가 없다."
        )
    limitations.append(
        "Adaptive attack을 평가하지 않았다. 공격자가 이 detector를 알고 있는 경우의 "
        "내성은 측정되지 않았다."
    )
    if clean_tar_delta_pp is None:
        limitations.append(
            "Clean 인증 성능에 미치는 영향(clean TAR delta)을 측정하지 않았다. "
            "07_DEFENSE_AND_DETECTION_SPEC.md 7절 기준으로 판정할 수 없다."
        )
    elif abs(clean_tar_delta_pp) > CLEAN_COST_BUDGET_PP:
        limitations.append(
            f"Clean TAR 감소 {abs(clean_tar_delta_pp):.2f}%p로 "
            f"07_DEFENSE_AND_DETECTION_SPEC.md 7절 예산 {CLEAN_COST_BUDGET_PP}%p를 초과한다."
        )
    if asr_reduction is None:
        limitations.append(
            "Conditional ASR 감소를 측정하지 않았다. 07 7절의 나머지 통과 기준(50% 이상 감소) "
            "판정에는 방어 전후 인증 결과 비교가 필요하다."
        )
    elif asr_reduction < ASR_REDUCTION_BUDGET:
        limitations.append(
            f"Conditional ASR 감소 {asr_reduction:.1%}로 07 7절 기준 "
            f"{ASR_REDUCTION_BUDGET:.0%}에 미치지 못한다."
        )
    return limitations


def per_attack_kind_tpr(rows, table, selected, threshold, window_frames):
    """
    공격 종류별 TPR. 07 7절은 공격 성공률을 단일 평균으로 숨기지 말라고 요구한다.

    종류별 표본 수가 적으면 점추정만으로 판단할 수 없으므로 분자와 분모를 함께 낸다.
    """
    import numpy as np

    from src.verification.defenses.probe_analyze import combine_clean_normalized

    # sample_id 순서대로 종류를 모은다. combine 결과와 같은 순서다.
    order, kinds = [], {}
    for row in rows:
        if row["label"] != "adversarial":
            continue
        if row["sample_id"] not in kinds:
            kinds[row["sample_id"]] = row["attack_kind"] or "unspecified"
            order.append(row["sample_id"])

    _, adversarial_scores = combine_clean_normalized(table, selected)
    aggregated = _aggregate(adversarial_scores, window_frames)

    # 집계 윈도는 연속 표본을 묶으므로 윈도의 종류는 시작 표본의 종류로 본다.
    result = {}
    for index, value in enumerate(aggregated):
        kind = kinds[order[index]] if index < len(order) else "unspecified"
        bucket = result.setdefault(kind, {"detected": 0, "total": 0})
        bucket["total"] += 1
        bucket["detected"] += int(value >= threshold)
    for kind, bucket in result.items():
        bucket["tpr"] = (
            bucket["detected"] / bucket["total"] if bucket["total"] else None
        )
    return result


def _aggregate(scores, window_frames: int):
    """연속 프레임을 윈도로 묶어 최악값을 취한다. face_auth 게이트와 같은 규칙이다."""
    import numpy as np

    scores = np.asarray(scores, float)
    if window_frames <= 1 or scores.size < window_frames:
        return scores
    return np.array(
        [scores[i : i + window_frames].max() for i in range(scores.size - window_frames + 1)]
    )


def build_artifact(
    probe_csv,
    sidecars,
    *,
    target_fpr: float = 0.01,
    top_k: int = 6,
    window_frames: int = 3,
    identity_threshold: float | None = None,
    artifact_id: str | None = None,
) -> dict:
    probe_csv = Path(probe_csv)
    rows = load_probe_rows(probe_csv)
    table = feature_table(rows)

    clean_rows = [row for row in rows if row["label"] == "clean"]
    n_clean = len({row["sample_id"] for row in clean_rows})
    if int(target_fpr * n_clean) < 1:
        raise InsufficientCleanSamplesError(
            f"clean 표본 {n_clean}개로는 목표 FPR {target_fpr}을 만족하는 임계값을 "
            f"관측값에서 고를 수 없다. 최소 {int(1 / target_fpr)}개가 필요하다."
        )

    ranked = sorted(
        (key for key in table if roc_auc(table[key]["adversarial"], table[key]["clean"]) is not None),
        key=lambda key: -roc_auc(table[key]["adversarial"], table[key]["clean"]),
    )
    selected = ranked[:top_k]
    clean_scores, adversarial_scores = combine_clean_normalized(table, selected)

    # face_auth 게이트는 최근 window_frames개 중 최악값을 쓴다. 프레임 단위로 정한
    # 임계값을 세션에 그대로 쓰면 실현 FPR이 윈도 크기만큼 배가된다. 적용 단위와
    # 같은 단위로 캘리브레이션한다.
    clean_windows = _aggregate(clean_scores, window_frames)
    adversarial_windows = _aggregate(adversarial_scores, window_frames)

    threshold = threshold_at_fpr(clean_windows, target_fpr)
    metrics = detector_metrics(
        clean=clean_windows, adversarial=adversarial_windows, threshold=threshold
    )
    clean_tar_delta_pp = (
        -metrics["fpr"] * 100.0 if metrics["fpr"] is not None else None
    )

    # 방어 전후 인증 비교. 신원 임계값이 주어질 때만 계산한다. 07 7절의 두 통과
    # 기준 중 conditional ASR 감소를 판정하려면 이 값이 필요하다.
    defense_comparison = None
    if identity_threshold is not None:
        sims = {}
        for row in rows:
            sims.setdefault(row["sample_id"], (row["label"], float(row["cos_orig_enroll"])))
        clean_sim = _aggregate(
            np.asarray([v for _, (l, v) in sims.items() if l == "clean"], float), window_frames
        )
        adversarial_sim = _aggregate(
            np.asarray([v for _, (l, v) in sims.items() if l == "adversarial"], float),
            window_frames,
        )
        try:
            attack_side = conditional_defense_metrics(
                attack_similarity=adversarial_sim,
                attack_detected=adversarial_windows >= threshold,
                identity_threshold=identity_threshold,
            )
        except NoEligibleAttackError as error:
            attack_side = {"error": str(error)}
        defense_comparison = {
            "attack": attack_side,
            "clean": clean_cost(
                clean_similarity=clean_sim,
                clean_detected=clean_windows >= threshold,
                identity_threshold=identity_threshold,
            ),
        }

    normalization = {}
    for transform, measure in selected:
        values = table[(transform, measure)]["clean"]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        normalization[f"{transform}|{measure}"] = {
            "clean_mean": round(mean, 8),
            "clean_std": round(variance ** 0.5, 8),
        }

    first = sidecars[0]
    subjects = {sidecar.get("subject_id") for sidecar in sidecars if sidecar.get("subject_id")}
    sessions = {sidecar.get("session_id") for sidecar in sidecars if sidecar.get("session_id")}
    attack_kinds = sorted(
        {sidecar.get("attack", {}).get("kind") for sidecar in sidecars if sidecar.get("attack", {}).get("kind")}
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "detector_threshold_artifact_id": artifact_id
        or f"det-{'-'.join(sorted(sessions))}-fpr{target_fpr}",
        "detector_id": "combined" if len(selected) > 1 else f"{selected[0][1]}",
        "feature": {
            "kind": "combined" if len(selected) > 1 else "single",
            "members": [
                {"transform": transform, "measure": measure}
                for transform, measure in selected
            ],
            "combination_rule": (
                "clean 통계로 z 정규화한 뒤 합산. 공격 라벨로 가중치를 학습하지 않는다."
                if len(selected) > 1
                else None
            ),
            "normalization": normalization,
        },
        "aggregation": {
            "unit": "session" if window_frames > 1 else "frame",
            "window_frames": window_frames,
            "rule": "max",
        },
        "score_direction": "higher_is_adversarial",
        "decision_rule": "score >= threshold 이면 hit",
        "threshold": round(float(threshold), 8),
        "selection_method": "target_fpr",
        "target_fpr": target_fpr,
        "model": first["model"],
        "transforms": first["transforms"],
        "calibration": {
            "session_ids": sorted(sessions),
            "subject_ids": sorted(subjects),
            "probe_csv_sha256": _sha256(probe_csv),
            "n_clean": n_clean,
            "achieved_fpr": metrics["fpr"],
            "n_clean_windows": int(len(clean_windows)),
            "jpeg_headroom_q75": first.get("jpeg_headroom_q75"),
        },
        "evaluation": {
            "n_adversarial": metrics["n_adversarial"],
            "attack_kinds": attack_kinds,
            "tpr": metrics["tpr"],
            "fpr": metrics["fpr"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "roc_auc": roc_auc(adversarial_scores, clean_scores),
            "true_positive": metrics["true_positive"],
            "false_negative": metrics["false_negative"],
            "false_positive": metrics["false_positive"],
            "true_negative": metrics["true_negative"],
            "tpr_by_attack_kind": per_attack_kind_tpr(
                rows, table, selected, threshold, window_frames
            ),
            "defense_comparison": defense_comparison,
            "clean_tar_delta_pp": (
                round(clean_tar_delta_pp, 4) if clean_tar_delta_pp is not None else None
            ),
            "meets_clean_cost_budget": (
                abs(clean_tar_delta_pp) <= CLEAN_COST_BUDGET_PP
                if clean_tar_delta_pp is not None
                else None
            ),
        },
        "limitations": derive_limitations(
            sidecars,
            subjects=subjects,
            sessions=sessions,
            clean_tar_delta_pp=clean_tar_delta_pp,
            asr_reduction=(
                defense_comparison["attack"].get("conditional_asr_reduction")
                if defense_comparison
                else None
            ),
        ),
        "created_by_run_id": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="detector threshold artifact 생성")
    parser.add_argument("--probe", required=True, help="probe.csv 경로")
    parser.add_argument("--session", required=True, action="append", help="session.json 경로. 반복 가능")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--window-frames",
        type=int,
        default=3,
        help="게이트가 묶는 프레임 수. face_auth FeatureSqueezeConfig.max_frames와 맞춘다",
    )
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=None,
        help="신원 임계값. 주면 07 7절의 conditional ASR 감소와 clean TAR delta를 판정한다",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sidecars = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.session]
    artifact = build_artifact(
        args.probe,
        sidecars,
        target_fpr=args.target_fpr,
        top_k=args.top_k,
        window_frames=args.window_frames,
        identity_threshold=args.identity_threshold,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"threshold {artifact['threshold']:.6f}  (목표 FPR {artifact['target_fpr']})")
    print(f"  달성 FPR {artifact['calibration']['achieved_fpr']}  clean {artifact['calibration']['n_clean']}")
    print(f"  집계 {artifact['aggregation']['unit']} (윈도 {artifact['aggregation']['window_frames']}, {artifact['aggregation']['rule']})")
    print(f"  clean TAR delta {artifact['evaluation']['clean_tar_delta_pp']}%p  "
          f"예산({CLEAN_COST_BUDGET_PP}%p) 충족 {artifact['evaluation']['meets_clean_cost_budget']}")
    for kind, bucket in sorted(artifact["evaluation"]["tpr_by_attack_kind"].items()):
        print(f"  공격 {kind:<14} TPR {bucket['tpr']:.4f}  ({bucket['detected']}/{bucket['total']})")
    print(f"  TPR {artifact['evaluation']['tpr']}  AUC {artifact['evaluation']['roc_auc']:.4f}  adv {artifact['evaluation']['n_adversarial']}")
    comparison = artifact["evaluation"].get("defense_comparison")
    if comparison and "error" not in comparison["attack"]:
        attack, clean = comparison["attack"], comparison["clean"]
        print()
        print("07 7절 잠정 통과 기준")
        print(f"  conditional ASR {attack['conditional_asr_before_defense']:.3f} → "
              f"{attack['conditional_asr_after_defense']:.3f} "
              f"(감소 {attack['conditional_asr_reduction']:.1%}, 기준 50%) "
              f"→ {'충족' if attack['meets_asr_budget'] else '미충족'}")
        print(f"  clean TAR {clean['clean_tar_before']:.4f} → {clean['clean_tar_after']:.4f} "
              f"({clean['clean_tar_delta_pp']:+.2f}%p, 기준 2%p) "
              f"→ {'충족' if clean['meets_clean_budget'] else '미충족'}")

    print(f"\n한계 {len(artifact['limitations'])}건:")
    for item in artifact["limitations"]:
        print(f"  - {item}")
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
