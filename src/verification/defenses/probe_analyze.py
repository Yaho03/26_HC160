"""
Probe 분석 — 계측 CSV에서 detector 지표와 임계값 후보를 산출한다.

지표 정의는 docs/09_EVALUATION_METRICS.md 1절과 4절을 따른다. 분모가 0이면 0이
아니라 undefined를 반환한다.

임계값은 clean 표본만으로 정한다. adversarial 표본은 TPR 측정에만 쓴다. 근거는
docs/07_DEFENSE_AND_DETECTION_SPEC.md 5절이다. 이 규칙을 코드로 강제하기 위해
threshold_at_fpr는 라벨이 섞인 입력을 거부한다.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MEASURES: tuple[str, ...] = ("self_consistency", "template_shift")


class ThresholdFromAttackDataError(ValueError):
    """임계값 산출에 adversarial 표본이 섞였다."""


def roc_auc(positive, negative) -> float | None:
    """Mann-Whitney U 기반 ROC-AUC. 한쪽이 비면 undefined."""
    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    if positive.size == 0 or negative.size == 0:
        return None

    values = np.concatenate([positive, negative])
    order = values.argsort(kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1)

    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    summed = np.zeros(counts.size)
    np.add.at(summed, inverse, ranks)
    ranks = (summed / counts)[inverse]

    rank_sum = ranks[: positive.size].sum()
    return float(
        (rank_sum - positive.size * (positive.size + 1) / 2)
        / (positive.size * negative.size)
    )


def threshold_at_fpr(clean_scores, target_fpr: float, labels=None) -> float | None:
    """clean 분위수로 임계값을 정한다. 라벨이 섞여 들어오면 거부한다."""
    if labels is not None and set(labels) - {"clean"}:
        raise ThresholdFromAttackDataError(
            "임계값은 clean 표본만으로 정한다. adversarial 행을 제외하고 호출하라."
        )
    scores = np.asarray(clean_scores, float)
    if scores.size == 0:
        return None

    # 목표 FPR을 넘지 않는 가장 낮은 관측값을 고른다. 분위수 선형 보간은 관측되지
    # 않은 값을 만들고 동점 처리에서 목표를 초과할 수 있다.
    #
    # target_fpr * n 이 1보다 작으면 어떤 관측값을 골라도 목표를 만족할 수 없다.
    # 표본이 부족한 것이므로 최댓값 위로 올려 FPR 0을 택하고, 그 사실은 표본 수와
    # 함께 보고한다.
    allowed = int(np.floor(target_fpr * scores.size))
    ordered = np.sort(scores)
    candidates = np.unique(ordered)
    counts = scores.size - np.searchsorted(ordered, candidates, side="left")
    feasible = np.flatnonzero(counts <= allowed)
    if feasible.size:
        return float(candidates[feasible[0]])
    return float(np.nextafter(candidates[-1], np.inf))


def detector_metrics(*, clean, adversarial, threshold: float) -> dict:
    """09 4절의 detector 지표. 값이 threshold 이상이면 탐지로 본다."""
    clean, adversarial = np.asarray(clean, float), np.asarray(adversarial, float)
    true_positive = int(np.sum(adversarial >= threshold))
    false_negative = int(adversarial.size - true_positive)
    false_positive = int(np.sum(clean >= threshold))
    true_negative = int(clean.size - false_positive)

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else None

    return {
        "threshold": float(threshold),
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "n_clean": int(clean.size),
        "n_adversarial": int(adversarial.size),
        "tpr": ratio(true_positive, adversarial.size),
        "fpr": ratio(false_positive, clean.size),
        "precision": ratio(true_positive, true_positive + false_positive),
        "recall": ratio(true_positive, adversarial.size),
    }


def load_probe_rows(path) -> list[dict]:
    """계측 CSV를 읽고 원시 코사인에서 두 측정량을 파생한다."""
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            cos_orig_enroll = float(raw["cos_orig_enroll"])
            cos_transformed_enroll = float(raw["cos_transformed_enroll"])
            cos_orig_transformed = float(raw["cos_orig_transformed"])
            rows.append(
                {
                    # attack_kind 컬럼이 없던 초기 세션은 미상으로 둔다. 소급 적용하지
                    # 않고, 종류별 보고에서 unspecified로 드러낸다.
                    "attack_kind": raw.get("attack_kind") or "",
                    **raw,
                    "attack_kind": raw.get("attack_kind") or (
                        "" if raw["label"] == "clean" else "unspecified"
                    ),
                    "self_consistency": 1.0 - cos_orig_transformed,
                    "template_shift": abs(cos_orig_enroll - cos_transformed_enroll),
                }
            )
    return rows


def feature_table(rows) -> dict:
    """(변환, 측정량) -> {clean: [...], adversarial: [...]}"""
    table: dict = defaultdict(lambda: {"clean": [], "adversarial": []})
    for row in rows:
        for measure in MEASURES:
            table[(row["transform"], measure)][row["label"]].append(row[measure])
    return dict(table)


def combine_clean_normalized(table: dict, keys) -> tuple[np.ndarray, np.ndarray]:
    """
    선택한 특징을 clean 통계로만 정규화해 더한다.

    가중치를 공격 라벨로 학습하지 않는다. 표본이 적을 때 과적합하고, clean만으로
    캘리브레이션한다는 규칙도 어긴다.
    """
    clean_total = adversarial_total = None
    for key in keys:
        clean = np.asarray(table[key]["clean"], float)
        adversarial = np.asarray(table[key]["adversarial"], float)
        mean = clean.mean() if clean.size else 0.0
        deviation = clean.std() if clean.size else 1.0
        deviation = deviation if deviation > 1e-12 else 1.0

        clean_z = (clean - mean) / deviation
        adversarial_z = (adversarial - mean) / deviation
        clean_total = clean_z if clean_total is None else clean_total + clean_z
        adversarial_total = (
            adversarial_z if adversarial_total is None else adversarial_total + adversarial_z
        )
    return clean_total, adversarial_total


def analyze(path, *, target_fpr: float = 0.01, top_k: int = 6) -> dict:
    rows = load_probe_rows(path)
    table = feature_table(rows)

    per_feature = {}
    for (transform, measure), values in table.items():
        auc = roc_auc(values["adversarial"], values["clean"])
        threshold = threshold_at_fpr(values["clean"], target_fpr)
        entry = {"roc_auc": auc}
        if threshold is not None:
            entry.update(
                detector_metrics(
                    clean=values["clean"],
                    adversarial=values["adversarial"],
                    threshold=threshold,
                )
            )
        per_feature[f"{transform}|{measure}"] = entry

    ranked = sorted(
        (key for key in table if per_feature[f"{key[0]}|{key[1]}"]["roc_auc"] is not None),
        key=lambda key: -per_feature[f"{key[0]}|{key[1]}"]["roc_auc"],
    )

    combined = {}
    if ranked:
        selected = ranked[:top_k]
        clean_scores, adversarial_scores = combine_clean_normalized(table, selected)
        threshold = threshold_at_fpr(clean_scores, target_fpr)
        combined = {
            "features": [f"{transform}|{measure}" for transform, measure in selected],
            "roc_auc": roc_auc(adversarial_scores, clean_scores),
            **(
                detector_metrics(
                    clean=clean_scores, adversarial=adversarial_scores, threshold=threshold
                )
                if threshold is not None
                else {}
            ),
        }

    return {
        "source": Path(path).name,
        "target_fpr": target_fpr,
        "n_rows": len(rows),
        "per_feature": per_feature,
        "ranked": [f"{transform}|{measure}" for transform, measure in ranked],
        "combined": combined,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="EXP-DET-001 계측 CSV 분석")
    parser.add_argument("--probe", required=True, help="probe.csv 경로")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--out", default=None, help="결과 JSON 경로")
    args = parser.parse_args()

    report = analyze(args.probe, target_fpr=args.target_fpr, top_k=args.top_k)

    print(f"행 {report['n_rows']}개, 목표 FPR {report['target_fpr']}\n")
    print(f"{'특징':<32} {'AUC':>8} {'TPR':>8} {'FPR':>8} {'n_clean':>8} {'n_adv':>6}")
    print("-" * 76)
    for name in report["ranked"]:
        entry = report["per_feature"][name]
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else "  n/a"
        print(
            f"{name:<32} {fmt(entry['roc_auc']):>8} {fmt(entry.get('tpr')):>8} "
            f"{fmt(entry.get('fpr')):>8} {entry.get('n_clean','?'):>8} {entry.get('n_adversarial','?'):>6}"
        )

    if report["combined"]:
        combined = report["combined"]
        print(f"\n결합 (상위 {len(combined['features'])}개, clean 통계로만 정규화)")
        print(f"  AUC {combined['roc_auc']:.4f}   TPR {combined.get('tpr')}   FPR {combined.get('fpr')}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
