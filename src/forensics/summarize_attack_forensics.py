"""Summarize FaceAuth attack-forensics outputs for dashboard and reports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.forensics.build_attack_forensics import (
    build_context,
    build_summary,
    risk_level,
    rule_findings,
    score_from_hits,
)
from src.forensics.privacy import sanitize_identity_and_paths


RISK_ORDER = ["critical", "high", "medium", "low"]


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: str, default: float = 0.0) -> float:
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def split_rules(value: str) -> list[str]:
    return [rule for rule in value.split(";") if rule]


def refresh_and_sanitize_sessions(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Recompute rule evidence, then remove direct identities and file paths."""
    context_rows = [
        dict(row)
        | {
            "source_name": row.get("source_identity", ""),
            "target_name": row.get("target_identity", ""),
        }
        for row in rows
    ]
    context = build_context(context_rows)
    sanitized_rows: list[dict[str, Any]] = []
    for original, context_row in zip(rows, context_rows):
        defense_state = {
            "defense_bypassed": parse_bool(original.get("defense_bypassed", "")),
            "strong_defense_bypassed": parse_bool(original.get("strong_defense_bypassed", "")),
            "bypassed_defenses": original.get("bypassed_defenses", ""),
        }
        margin = parse_float(original.get("threshold_margin", ""))
        findings = rule_findings(context_row, defense_state, context, margin)
        hits = [finding["rule_id"] for finding in findings]
        accepted = parse_bool(original.get("accepted_after_attack", ""))
        score = score_from_hits(
            hits,
            accepted,
            defense_state["defense_bypassed"],
            defense_state["strong_defense_bypassed"],
        )
        refreshed: dict[str, Any] = dict(original)
        refreshed.update(
            {
                "risk_score": score,
                "risk_level": risk_level(score),
                "rule_hits": ";".join(hits),
                "rule_reasons": json.dumps(findings, ensure_ascii=False, sort_keys=True),
            }
        )
        sanitized_rows.append(sanitize_identity_and_paths(refreshed))
    return sanitized_rows


def build_overview(rows: list[dict[str, str]]) -> dict[str, Any]:
    total = len(rows)
    accepted = sum(parse_bool(row["accepted_after_attack"]) for row in rows)
    high_or_critical = sum(row["risk_level"] in {"high", "critical"} for row in rows)
    critical = sum(row["risk_level"] == "critical" for row in rows)
    families = Counter(row["attack_family"] for row in rows)
    risks = Counter(row["risk_level"] for row in rows)
    avg_risk = round(
        sum(parse_float(row["risk_score"]) for row in rows) / total,
        3,
    ) if total else 0.0

    return {
        "total_sessions": total,
        "accepted_after_attack": accepted,
        "attack_accept_rate": pct(accepted, total),
        "critical_sessions": critical,
        "high_or_critical_sessions": high_or_critical,
        "high_or_critical_rate": pct(high_or_critical, total),
        "avg_risk_score": avg_risk,
        "attack_family_counts": dict(sorted(families.items())),
        "risk_level_counts": {risk: risks.get(risk, 0) for risk in RISK_ORDER},
    }


def build_family_matrix(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_family[row["attack_family"]].append(row)

    matrix = []
    for family in sorted(by_family):
        family_rows = by_family[family]
        accepted = sum(parse_bool(row["accepted_after_attack"]) for row in family_rows)
        risks = Counter(row["risk_level"] for row in family_rows)
        matrix.append({
            "attack_family": family,
            "sessions": len(family_rows),
            "accepted_after_attack": accepted,
            "attack_accept_rate": pct(accepted, len(family_rows)),
            "critical": risks.get("critical", 0),
            "high": risks.get("high", 0),
            "medium": risks.get("medium", 0),
            "low": risks.get("low", 0),
            "avg_risk_score": round(
                sum(parse_float(row["risk_score"]) for row in family_rows) / len(family_rows),
                3,
            ),
        })
    return matrix


def build_rule_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    success_counts: Counter[str] = Counter()
    for row in rows:
        rules = split_rules(row["rule_hits"])
        for rule in rules:
            counts[rule] += 1
            if parse_bool(row["accepted_after_attack"]):
                success_counts[rule] += 1

    return [
        {
            "rule_id": rule,
            "sessions": count,
            "accepted_after_attack": success_counts[rule],
            "attack_accept_rate": pct(success_counts[rule], count),
        }
        for rule, count in counts.most_common()
    ]


def build_top_risk(rows: list[dict[str, str]], limit: int) -> list[dict[str, Any]]:
    selected_columns = [
        "session_id",
        "timestamp",
        "account_id",
        "source_identity",
        "target_identity",
        "attack_family",
        "epsilon",
        "similarity_before",
        "similarity_after_attack",
        "threshold_margin",
        "accepted_after_attack",
        "risk_score",
        "risk_level",
        "rule_hits",
        "adv_file",
    ]
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            parse_float(row["risk_score"]),
            parse_float(row["threshold_margin"]),
            parse_float(row["similarity_after_attack"]),
        ),
        reverse=True,
    )
    return [
        {column: row.get(column, "") for column in selected_columns}
        for row in sorted_rows[:limit]
    ]


def build_markdown(
    overview: dict[str, Any],
    family_matrix: list[dict[str, Any]],
    rule_summary: list[dict[str, Any]],
) -> str:
    lines = [
        "# FaceAuth Attack Forensics 결과 요약",
        "",
        "작성일: 2026-06-29",
        "",
        "## 핵심 결과",
        "",
        f"- 총 공격 세션: {overview['total_sessions']:,}",
        f"- 공격 후 인증 성공: {overview['accepted_after_attack']:,} ({overview['attack_accept_rate'] * 100:.2f}%)",
        f"- high/critical 위험 세션: {overview['high_or_critical_sessions']:,} ({overview['high_or_critical_rate'] * 100:.2f}%)",
        f"- critical 세션: {overview['critical_sessions']:,}",
        f"- 평균 risk score: {overview['avg_risk_score']}",
        "",
        "## 공격 유형별 결과",
        "",
        "| attack_family | sessions | accepted | accept_rate | critical | high | medium | low | avg_risk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in family_matrix:
        lines.append(
            "| {attack_family} | {sessions} | {accepted_after_attack} | {rate:.2f}% | "
            "{critical} | {high} | {medium} | {low} | {avg_risk_score} |".format(
                rate=row["attack_accept_rate"] * 100,
                **row,
            )
        )

    lines.extend([
        "",
        "## 탐지 룰 상위 분포",
        "",
        "| rule_id | sessions | accepted | accept_rate |",
        "|---|---:|---:|---:|",
    ])
    for row in rule_summary:
        lines.append(
            "| {rule_id} | {sessions} | {accepted_after_attack} | {rate:.2f}% |".format(
                rate=row["attack_accept_rate"] * 100,
                **row,
            )
        )

    lines.extend([
        "",
        "## 해석",
        "",
        "- PGD와 FGSM은 낮은 Linf 예산에서도 공격 후 accept 비율이 높아 white-box 계열 위험도를 대표한다.",
        "- Square는 query 비용이 높지만 이번 설정에서는 성공률이 낮아, 대시보드에서는 high-query probing 징후로 보여주는 편이 적합하다.",
        "- Adaptive smoothing 공격은 성공률이 낮지만 방어 인지 공격 시나리오를 설명하는 근거로 남긴다.",
        "- 현재 결과는 실제 금융 로그가 아니라 LFW/FaceNet 실험 metadata 기반 모의 세션이다.",
        "",
        "## 대시보드 전달 파일",
        "",
        "- `outputs/forensics/attack_sessions.csv`: 세션 단위 상세 로그",
        "- `outputs/forensics/attack_risk_summary.csv`: overview 카드용 요약",
        "- `outputs/forensics/attack_detection_rules.json`: 탐지 룰 설명",
        "- `outputs/forensics/dashboard_overview.json`: 대시보드 초기 카드/차트용 요약",
        "- `outputs/forensics/top_risk_sessions.csv`: 상세 화면에 바로 띄울 high-priority 세션",
    ])
    return "\n".join(lines) + "\n"


def write_plots(
    rows: list[dict[str, Any]],
    family_matrix: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    """Create deterministic, identity-free dashboard plots."""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "hc160-forensics-v1"
    import matplotlib.pyplot as plt

    families = sorted({str(row["attack_family"]) for row in rows})
    palette = plt.get_cmap("tab10")
    colors = {family: palette(index % 10) for index, family in enumerate(families)}

    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for family in families:
        selected = [row for row in rows if row["attack_family"] == family]
        axis.scatter(
            [parse_float(str(row["similarity_before"])) for row in selected],
            [parse_float(str(row["similarity_after_attack"])) for row in selected],
            s=12,
            alpha=0.45,
            color=colors[family],
            label=family,
        )
    thresholds = sorted({parse_float(str(row["threshold"])) for row in rows})
    for threshold in thresholds:
        axis.axhline(threshold, color="#d62728", linewidth=1, linestyle="--", alpha=0.55)
    axis.plot([-1, 1], [-1, 1], color="#666666", linewidth=1, linestyle=":")
    axis.set(xlabel="Cosine similarity before attack", ylabel="Cosine similarity after attack")
    axis.set_title("Attack similarity shift and verification thresholds")
    axis.legend(title="Attack family", fontsize=8)
    figure.savefig(out_dir / "attack_similarity_panel.png", dpi=160, metadata={"Software": "HC160"})
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    labels = [str(row["attack_family"]) for row in family_matrix]
    accept_rates = [float(row["attack_accept_rate"]) * 100 for row in family_matrix]
    bars = axis.bar(labels, accept_rates, color=[colors[label] for label in labels])
    axis.set(ylim=(0, 100), ylabel="Accepted after attack (%)")
    axis.set_title("Targeted attack acceptance rate by family")
    axis.bar_label(bars, labels=[f"{value:.1f}%" for value in accept_rates], padding=3, fontsize=8)
    figure.savefig(out_dir / "attack_family_overview.png", dpi=160, metadata={"Software": "HC160"})
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=Path, default=Path("outputs/forensics/attack_sessions.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/forensics"))
    parser.add_argument("--top-limit", type=int, default=50)
    parser.add_argument("--markdown", type=Path, default=Path("docs/faceauth_attack_forensics_results_2026-06-29.md"))
    args = parser.parse_args()

    rows = refresh_and_sanitize_sessions(read_csv(args.sessions))
    overview = build_overview(rows)
    family_matrix = build_family_matrix(rows)
    rule_summary = build_rule_summary(rows)
    top_risk = build_top_risk(rows, args.top_limit)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "attack_sessions.csv", rows)
    summary_rows = [
        dict(row)
        | {
            "accepted_after_attack": parse_bool(str(row.get("accepted_after_attack", ""))),
            "defense_bypassed": parse_bool(str(row.get("defense_bypassed", ""))),
        }
        for row in rows
    ]
    write_csv(args.out_dir / "attack_risk_summary.csv", build_summary(summary_rows))
    (args.out_dir / "dashboard_overview.json").write_text(
        json.dumps(overview, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.out_dir / "attack_family_matrix.csv", family_matrix)
    write_csv(args.out_dir / "rule_hit_summary.csv", rule_summary)
    write_csv(args.out_dir / "top_risk_sessions.csv", top_risk)
    write_plots(rows, family_matrix, args.out_dir)

    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(
        build_markdown(overview, family_matrix, rule_summary),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
