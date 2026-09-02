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


class InsufficientCleanSamplesError(ValueError):
    """clean 표본 수가 목표 FPR을 만족할 수 없다."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_limitations(sidecars, *, subjects, sessions) -> list[str]:
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
    limitations.append(
        "Clean 인증 성능에 미치는 영향(clean TAR delta)을 측정하지 않았다. "
        "07_DEFENSE_AND_DETECTION_SPEC.md 7절 기준으로 판정할 수 없다."
    )
    return limitations


def build_artifact(
    probe_csv,
    sidecars,
    *,
    target_fpr: float = 0.01,
    top_k: int = 6,
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
    threshold = threshold_at_fpr(clean_scores, target_fpr)
    metrics = detector_metrics(
        clean=clean_scores, adversarial=adversarial_scores, threshold=threshold
    )

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
        },
        "limitations": derive_limitations(sidecars, subjects=subjects, sessions=sessions),
        "created_by_run_id": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="detector threshold artifact 생성")
    parser.add_argument("--probe", required=True, help="probe.csv 경로")
    parser.add_argument("--session", required=True, action="append", help="session.json 경로. 반복 가능")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sidecars = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.session]
    artifact = build_artifact(
        args.probe, sidecars, target_fpr=args.target_fpr, top_k=args.top_k
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"threshold {artifact['threshold']:.6f}  (목표 FPR {artifact['target_fpr']})")
    print(f"  달성 FPR {artifact['calibration']['achieved_fpr']}  clean {artifact['calibration']['n_clean']}")
    print(f"  TPR {artifact['evaluation']['tpr']}  AUC {artifact['evaluation']['roc_auc']:.4f}  adv {artifact['evaluation']['n_adversarial']}")
    print(f"\n한계 {len(artifact['limitations'])}건:")
    for item in artifact["limitations"]:
        print(f"  - {item}")
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
