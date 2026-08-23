"""Audit and summarize a defense handoff without mixing evaluation bases."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.forensics.privacy import sanitize_identity_and_paths


PREPROCESSING = {"jpeg", "smoothing", "bitdepth"}
DEFENSES = ("jpeg", "smoothing", "bitdepth", "adv_training")
EXPECTED_EVALUATION_BASIS = {
    "jpeg": "artifact_reload",
    "smoothing": "artifact_reload",
    "bitdepth": "artifact_reload",
    "adv_training": "legacy_record",
}
REQUIRED_DEFENSE_FIELDS = {
    "sample_id",
    "defense",
    "evaluation_basis",
    "attack_success_before_defense",
    "defense_bypassed",
    "defense_success",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def truthy(value: str) -> bool:
    return value.strip().lower() == "true"


def rate(count: int, total: int) -> str:
    return f"{count / total:.6f}" if total else ""


def summarize(rows: list[dict[str, str]], defenses: set[str]) -> list[dict[str, str | int]]:
    result = []
    for defense in sorted(defenses):
        selected = [row for row in rows if row["defense"] == defense]
        eligible = [row for row in selected if truthy(row["attack_success_before_defense"])]
        bypassed = sum(truthy(row["defense_bypassed"]) for row in eligible)
        succeeded = sum(truthy(row["defense_success"]) for row in eligible)
        result.append(
            {
                "defense": defense,
                "evaluation_basis": ";".join(sorted({row["evaluation_basis"] for row in selected})),
                "total_rows": len(selected),
                "eligible_attacks": len(eligible),
                "defense_bypassed": bypassed,
                "defense_bypass_rate": rate(bypassed, len(eligible)),
                "defense_success": succeeded,
                "defense_success_rate": rate(succeeded, len(eligible)),
            }
        )
    return result


def build_session_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_sample: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(row)

    sessions = []
    for sample_id, group in sorted(by_sample.items()):
        first = group[0]
        similarity = float(first.get("attack_index_similarity_after_attack") or 0)
        threshold = float(first.get("threshold") or 0)
        session: dict[str, object] = {
            "session_id": f"faceauth_defense_{sample_id.removeprefix('vf_')}",
            "attempt_id": sample_id,
            "pair_id": first.get("pair_id", ""),
            "attack": first.get("attack", ""),
            "attack_family": "pgd" if "pgd" in first.get("attack", "").lower() else "unknown",
            "source_identity": first.get("source_name", ""),
            "target_identity": first.get("target_name", ""),
            "epsilon": first.get("epsilon", ""),
            "alpha": first.get("alpha", ""),
            "steps": first.get("steps", ""),
            "threshold": first.get("threshold", ""),
            "similarity_before": first.get("similarity_before", ""),
            "similarity_after_attack": first.get("attack_index_similarity_after_attack", ""),
            "threshold_margin": round(similarity - threshold, 6),
            "accepted_after_attack": truthy(first.get("attack_index_accepted_after_attack", "")),
            "attack_success_before_defense": truthy(
                first.get("attack_index_attack_success_before_defense", "")
            ),
            "source_file": first.get("source_file", ""),
            "target_enroll_file": first.get("target_enroll_file", ""),
            "adv_file": first.get("adv_file", ""),
            "perturbation_file": first.get("perturbation_file", ""),
        }
        for row in group:
            prefix = row["defense"]
            session.update(
                {
                    f"{prefix}_evaluation_basis": row.get("evaluation_basis", ""),
                    f"{prefix}_attack_success_before_defense": truthy(
                        row.get("attack_success_before_defense", "")
                    ),
                    f"{prefix}_similarity_after_defense": row.get("similarity_after_defense", ""),
                    f"{prefix}_accepted_after_defense": truthy(row.get("accepted_after_defense", "")),
                    f"{prefix}_defense_bypassed": truthy(row.get("defense_bypassed", "")),
                    f"{prefix}_defense_success": truthy(row.get("defense_success", "")),
                    f"{prefix}_defense_time_sec": row.get("defense_time_sec", ""),
                }
            )
        sessions.append(sanitize_identity_and_paths(session))
    return sessions


def build_overview(
    rows: list[dict[str, str]],
    preprocessing: list[dict[str, str | int]],
    adversarial_training: list[dict[str, str | int]],
) -> dict[str, object]:
    def metrics(items: list[dict[str, str | int]]) -> dict[str, object]:
        def optional_rate(value: str | int) -> float | None:
            return float(str(value)) if str(value) else None

        return {
            str(item["defense"]): {
                "eligible_attacks": item["eligible_attacks"],
                "defense_bypassed": item["defense_bypassed"],
                "defense_bypass_rate": optional_rate(item["defense_bypass_rate"]),
                "defense_success": item["defense_success"],
                "defense_success_rate": optional_rate(item["defense_success_rate"]),
            }
            for item in items
        }

    return {
        "schema_version": 1,
        "cohort": "defense_evaluation",
        "sessions": len({row["sample_id"] for row in rows}),
        "result_rows": len(rows),
        "evaluation_groups": {
            "preprocessing": {
                "evaluation_basis": "artifact_reload",
                "denominator": "attack_success_before_defense=true per preprocessing defense",
                "defenses": metrics(preprocessing),
            },
            "adversarial_training": {
                "evaluation_basis": "legacy_record",
                "denominator": "attack_success_before_defense=true in adversarial-training evaluation",
                "defenses": metrics(adversarial_training),
            },
        },
        "combined_bypass_rate": None,
        "note": "Evaluation groups use different bases and must not be combined.",
    }


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def attack_config(row: dict[str, str]) -> tuple[str, str, str, str] | None:
    epsilon_match = re.search(r"/([0-9.]+)/vf_", row.get("adv_file", ""))
    steps_match = re.search(r"_s(\d+)\.", row.get("adv_file", ""))
    epsilon = row.get("epsilon") or (epsilon_match.group(1) if epsilon_match else "")
    steps = row.get("steps") or (steps_match.group(1) if steps_match else "")
    attack = row.get("attack_id") or row.get("attack", "")
    if not epsilon or not steps or not attack:
        return None
    return row.get("pair_id", ""), attack, str(float(epsilon)), steps


def audit(defense_rows: list[dict[str, str]], session_rows: list[dict[str, str]]) -> dict[str, object]:
    defense_ids = {row["sample_id"] for row in defense_rows}
    session_ids = {row["attempt_id"] for row in session_rows}
    matched_ids = defense_ids & session_ids

    sessions_by_config: defaultdict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for row in session_rows:
        sessions_by_config[(row["pair_id"], row["attack"], row["epsilon"], row["steps"])].append(
            row["attempt_id"]
        )

    candidates: Counter[int] = Counter()
    for row in {item["sample_id"]: item for item in defense_rows}.values():
        config = attack_config(row)
        candidates[len(sessions_by_config.get(config, [])) if config else 0] += 1

    self_contained = {
        "pair_id",
        "attack",
        "source_name",
        "target_name",
        "epsilon",
        "steps",
        "attack_index_similarity_after_attack",
    }.issubset(defense_rows[0])

    return {
        "aggregation_status": "ready_from_self_contained_handoff" if self_contained else "ready_from_defense_only",
        "handoff_contains_attack_index_columns": self_contained,
        "join_status": "ready" if len(matched_ids) == len(defense_ids) else "blocked_id_mismatch",
        "session_rows": len(session_rows),
        "unique_session_attempt_ids": len(session_ids),
        "defense_rows": len(defense_rows),
        "unique_defense_sample_ids": len(defense_ids),
        "exact_matched_sample_ids": len(matched_ids),
        "unmatched_defense_sample_ids": len(defense_ids - session_ids),
        "config_candidate_count_distribution": {str(key): value for key, value in sorted(candidates.items())},
        "unmatched_examples": sorted(defense_ids - session_ids)[:10],
        "note": "Config candidates are diagnostic only and are not safe replacements for sample_id joins.",
    }


def validate(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Defense handoff is empty")
    missing = REQUIRED_DEFENSE_FIELDS - rows[0].keys()
    if missing:
        raise ValueError(f"Missing defense columns: {sorted(missing)}")
    duplicate_keys = [key for key, count in Counter((r["sample_id"], r["defense"]) for r in rows).items() if count > 1]
    if duplicate_keys:
        raise ValueError(f"Duplicate sample/defense keys: {duplicate_keys[:3]}")
    defenses_by_sample: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        defense = row["defense"]
        if defense not in EXPECTED_EVALUATION_BASIS:
            raise ValueError(f"Unknown defense: {defense}")
        expected_basis = EXPECTED_EVALUATION_BASIS[defense]
        if row["evaluation_basis"] != expected_basis:
            raise ValueError(
                f"{row['sample_id']}/{defense}: evaluation_basis must be {expected_basis}, "
                f"got {row['evaluation_basis']}"
            )
        defenses_by_sample[row["sample_id"]].add(row["defense"])
    incomplete = [sample_id for sample_id, defenses in defenses_by_sample.items() if defenses != set(DEFENSES)]
    if incomplete:
        raise ValueError(f"Samples missing defense rows: {incomplete[:3]}")


def self_check() -> None:
    rows = [
        {"defense": "jpeg", "evaluation_basis": "artifact_reload", "attack_success_before_defense": "true", "defense_bypassed": "true", "defense_success": "false"},
        {"defense": "jpeg", "evaluation_basis": "artifact_reload", "attack_success_before_defense": "false", "defense_bypassed": "false", "defense_success": "false"},
    ]
    assert summarize(rows, {"jpeg"})[0]["defense_bypass_rate"] == "1.000000"
    session = build_session_rows(
        [
            rows[0]
            | {
                "sample_id": "vf_demo",
                "threshold": "0.5",
                "attack_index_similarity_after_attack": "0.6",
                "attack_index_accepted_after_attack": "true",
                "attack_index_attack_success_before_defense": "true",
                "accepted_after_defense": "true",
            }
        ]
    )[0]
    assert session["threshold_margin"] == 0.1 and session["jpeg_defense_bypassed"] is True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--defense-csv", type=Path)
    parser.add_argument("--sessions-csv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    if not all((args.defense_csv, args.sessions_csv, args.out_dir)):
        parser.error("--defense-csv, --sessions-csv, and --out-dir are required")

    defense_rows = read_csv(args.defense_csv)
    session_rows = read_csv(args.sessions_csv)
    validate(defense_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    preprocessing = summarize(defense_rows, PREPROCESSING)
    adversarial_training = summarize(defense_rows, {"adv_training"})
    published_defense_rows = [sanitize_identity_and_paths(row) for row in defense_rows]
    write_csv(args.out_dir / "defense_results_by_sample_id.csv", published_defense_rows)
    write_csv(args.out_dir / "defense_evaluation_sessions.csv", build_session_rows(defense_rows))
    write_csv(args.out_dir / "preprocessing_defense_summary.csv", preprocessing)
    write_csv(args.out_dir / "adversarial_training_defense_summary.csv", adversarial_training)
    (args.out_dir / "defense_integration_overview.json").write_text(
        json.dumps(
            build_overview(defense_rows, preprocessing, adversarial_training),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "defense_join_audit.json").write_text(
        json.dumps(audit(defense_rows, session_rows), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
