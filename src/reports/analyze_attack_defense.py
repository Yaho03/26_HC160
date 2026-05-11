"""Join attack metadata with per-sample defense results and summarize them.

This script intentionally uses only the Python standard library so it can run in
lightweight local environments as well as Colab.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DEFENSE_RESULT_NAMES = {
    "jpeg_results_q75.csv",
    "smoothing_results_r3p0.csv",
    "bitdepth_results_4bit.csv",
}


def str_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def str_float(value: object) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_defense_csvs(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*.csv") if p.name in DEFENSE_RESULT_NAMES)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_joined_rows(attack_rows: list[dict[str, str]], defense_files: list[Path]) -> list[dict[str, object]]:
    attack_by_id = {row["sample_id"]: row for row in attack_rows}
    joined: list[dict[str, object]] = []

    for result_file in defense_files:
        for defense in read_csv(result_file):
            sample_id = defense.get("sample_id", "")
            attack = attack_by_id.get(sample_id)
            attack_matched = attack is not None
            if attack is None:
                attack = {}

            attack_success_after = str_bool(defense.get("attack_success_after_defense"))
            recovered = str_bool(defense.get("recovered"))
            before = str_float(defense.get("target_conf_before_defense"))
            after = str_float(defense.get("target_conf_after_defense"))

            joined.append({
                "sample_id": sample_id,
                "attack_index_matched": attack_matched,
                "attack_family": attack.get("attack_family", defense.get("attack_family", "")),
                "attack": attack.get("attack", defense.get("attack", "")),
                "defense": defense.get("defense", ""),
                "defense_params": defense.get("defense_params", ""),
                "true_label": attack.get("true_label", defense.get("true_label", "")),
                "true_name": attack.get("true_name", ""),
                "target_label": attack.get("target_label", defense.get("target_label", "")),
                "target_name": attack.get("target_name", ""),
                "pred_before_name": attack.get("pred_before_name", ""),
                "pred_after_name": attack.get("pred_after_name", ""),
                "pred_after_defense": defense.get("pred_after_defense", ""),
                "pred_after_defense_name": defense.get("pred_after_defense_name", ""),
                "clean_correct": attack.get("clean_correct", ""),
                "success_on_clean": attack.get("success_on_clean", ""),
                "attack_success_after_defense": attack_success_after,
                "defense_success": not attack_success_after,
                "recovered": recovered,
                "attack_l0": attack.get("l0", ""),
                "attack_l2": attack.get("l2", ""),
                "attack_linf": attack.get("linf", ""),
                "attack_time_sec": attack.get("time_sec", ""),
                "queries_used": attack.get("queries_used", ""),
                "target_conf_before_attack": attack.get("target_conf_before", ""),
                "target_conf_after_attack": attack.get("target_conf_after", ""),
                "target_conf_before_defense": before,
                "target_conf_after_defense": after,
                "target_conf_drop": before - after,
                "defense_time_sec": str_float(defense.get("defense_time_sec")),
                "input_adv_file": defense.get("input_adv_file", ""),
                "defended_file": defense.get("defended_file", ""),
                "source_defense_file": str(result_file),
            })

    return joined


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    all_by_defense: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row in rows:
        defense = str(row["defense"])
        family = str(row["attack_family"])
        grouped[(defense, family)].append(row)
        all_by_defense[defense].append(row)

    summary: list[dict[str, object]] = []
    for defense, rows_for_defense in all_by_defense.items():
        grouped[(defense, "ALL")] = rows_for_defense

    for (defense, family), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        n = len(group)
        defense_successes = [1.0 if row["defense_success"] else 0.0 for row in group]
        recoveries = [1.0 if row["recovered"] else 0.0 for row in group]
        drops = [float(row["target_conf_drop"]) for row in group]
        defense_times = [float(row["defense_time_sec"]) for row in group]
        matched_attack_rows = [row for row in group if row["attack_index_matched"]]
        attack_l2 = [str_float(row["attack_l2"]) for row in matched_attack_rows if row["attack_l2"] != ""]
        attack_linf = [str_float(row["attack_linf"]) for row in matched_attack_rows if row["attack_linf"] != ""]

        summary.append({
            "defense": defense,
            "attack_family": family,
            "samples": n,
            "attack_index_matched": len(matched_attack_rows),
            "defense_success_rate": avg(defense_successes),
            "recovery_rate": avg(recoveries),
            "avg_target_conf_drop": avg(drops),
            "avg_defense_time_sec": avg(defense_times),
            "avg_attack_l2": avg(attack_l2) if attack_l2 else "",
            "avg_attack_linf": avg(attack_linf) if attack_linf else "",
        })

    return summary


def best_by_attack(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        family = str(row["attack_family"])
        if family != "ALL":
            by_family[family].append(row)

    best: list[dict[str, object]] = []
    for family, rows in sorted(by_family.items()):
        picked = max(
            rows,
            key=lambda row: (
                float(row["defense_success_rate"]),
                float(row["recovery_rate"]),
                float(row["avg_target_conf_drop"]),
            ),
        )
        best.append({
            "attack_family": family,
            "best_defense": picked["defense"],
            "samples": picked["samples"],
            "defense_success_rate": picked["defense_success_rate"],
            "recovery_rate": picked["recovery_rate"],
            "avg_target_conf_drop": picked["avg_target_conf_drop"],
        })
    return best


def write_report(path: Path, summary_rows: list[dict[str, object]], best_rows: list[dict[str, object]]) -> None:
    overall = [row for row in summary_rows if row["attack_family"] == "ALL"]
    attack_rows = [row for row in summary_rows if row["attack_family"] != "ALL"]

    lines = [
        "# Attack-Defense Integrated Analysis",
        "",
        "## Overall Defense Results",
        "",
        "| Defense | Samples | Defense Success | Recovery | Avg Target Conf Drop | Avg Time Sec |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(overall, key=lambda r: float(r["defense_success_rate"]), reverse=True):
        lines.append(
            f"| {row['defense']} | {row['samples']} | {pct(float(row['defense_success_rate']))} | "
            f"{pct(float(row['recovery_rate']))} | {float(row['avg_target_conf_drop']):.4f} | "
            f"{float(row['avg_defense_time_sec']):.4f} |"
        )

    lines.extend([
        "",
        "## Best Defense By Attack",
        "",
        "| Attack | Best Defense | Samples | Defense Success | Recovery | Avg Target Conf Drop |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in best_rows:
        lines.append(
            f"| {row['attack_family']} | {row['best_defense']} | {row['samples']} | "
            f"{pct(float(row['defense_success_rate']))} | {pct(float(row['recovery_rate']))} | "
            f"{float(row['avg_target_conf_drop']):.4f} |"
        )

    lines.extend([
        "",
        "## Full Attack x Defense Table",
        "",
        "| Defense | Attack | Samples | Defense Success | Recovery | Avg Target Conf Drop | Avg Time Sec |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(attack_rows, key=lambda r: (str(r["attack_family"]), str(r["defense"]))):
        lines.append(
            f"| {row['defense']} | {row['attack_family']} | {row['samples']} | "
            f"{pct(float(row['defense_success_rate']))} | {pct(float(row['recovery_rate']))} | "
            f"{float(row['avg_target_conf_drop']):.4f} | {float(row['avg_defense_time_sec']):.4f} |"
        )

    lines.extend([
        "",
        "## Interpretation Notes",
        "",
        "- Defense Success means the defended image is no longer classified as the attack target.",
        "- Recovery means the defended image is classified back as the original true identity.",
        "- Defense Success can be high while Recovery remains moderate, because the prediction may move to a third class.",
        "- ZOO has relatively few successful clean attack samples, so its percentages should be interpreted as reference values.",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze joined attack and defense results.")
    parser.add_argument("--attack-index", type=Path, default=Path("outputs/attacks/attack_index.csv"))
    parser.add_argument("--defense-root", type=Path, default=Path("outputs/defenses"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/analysis"))
    args = parser.parse_args()

    if not args.attack_index.exists():
        raise FileNotFoundError(f"attack_index.csv not found: {args.attack_index}")

    defense_files = find_defense_csvs(args.defense_root)
    if not defense_files:
        raise FileNotFoundError(f"No per-sample defense result CSVs found under {args.defense_root}")

    attack_rows = read_csv(args.attack_index)
    joined_rows = build_joined_rows(attack_rows, defense_files)
    if not joined_rows:
        raise ValueError("No rows joined. Check sample_id alignment between attack and defense results.")

    joined_fields = list(joined_rows[0].keys())
    summary_rows = summarize(joined_rows)
    best_rows = best_by_attack(summary_rows)

    summary_fields = [
        "defense",
        "attack_family",
        "samples",
        "attack_index_matched",
        "defense_success_rate",
        "recovery_rate",
        "avg_target_conf_drop",
        "avg_defense_time_sec",
        "avg_attack_l2",
        "avg_attack_linf",
    ]
    best_fields = [
        "attack_family",
        "best_defense",
        "samples",
        "defense_success_rate",
        "recovery_rate",
        "avg_target_conf_drop",
    ]

    write_csv(args.out_dir / "attack_defense_joined.csv", joined_rows, joined_fields)
    write_csv(args.out_dir / "attack_defense_summary.csv", summary_rows, summary_fields)
    write_csv(args.out_dir / "best_defense_by_attack.csv", best_rows, best_fields)
    write_report(args.out_dir / "attack_defense_report.md", summary_rows, best_rows)

    matched = sum(1 for row in joined_rows if row["attack_index_matched"])
    print(f"Defense files: {len(defense_files)}")
    for path in defense_files:
        print(f"- {path}")
    print(f"Joined rows: {len(joined_rows)}")
    print(f"Rows matched with attack_index: {matched}")
    if matched < len(joined_rows):
        print("Warning: Some defense rows were summarized without full attack_index metadata.")
    print(f"Saved: {args.out_dir / 'attack_defense_joined.csv'}")
    print(f"Saved: {args.out_dir / 'attack_defense_summary.csv'}")
    print(f"Saved: {args.out_dir / 'best_defense_by_attack.csv'}")
    print(f"Saved: {args.out_dir / 'attack_defense_report.md'}")


if __name__ == "__main__":
    main()
