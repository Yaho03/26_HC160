"""Create a defense-result template for FaceNet verification attack handoff rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


OUTPUT_COLUMNS = [
    "sample_id",
    "attack",
    "defense",
    "defense_params",
    "model",
    "pretrained",
    "source_file",
    "target_enroll_file",
    "adv_file",
    "defended_file",
    "source_name",
    "target_name",
    "threshold",
    "similarity_before",
    "similarity_after_attack",
    "similarity_after_defense",
    "accepted_before",
    "accepted_after_attack",
    "accepted_after_defense",
    "attack_success_before_defense",
    "attack_success_after_defense",
    "defense_success",
    "epsilon",
    "alpha",
    "steps",
    "l2",
    "linf",
    "defense_time_sec",
    "status",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create FaceNet verification defense result template.")
    parser.add_argument("--handoff-index", type=Path, default=Path("outputs/handoff/facenet_verification_attack_package/attack_handoff_index.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/handoff/facenet_verification_defense_results_template.csv"))
    args = parser.parse_args()

    rows = read_rows(args.handoff_index)
    if not rows:
        raise ValueError(f"No rows found in {args.handoff_index}")

    output_rows = []
    for row in rows:
        output_rows.append({
            "sample_id": row.get("sample_id", ""),
            "attack": row.get("attack", ""),
            "defense": "",
            "defense_params": "",
            "model": row.get("model", ""),
            "pretrained": row.get("pretrained", ""),
            "source_file": row.get("source_file", ""),
            "target_enroll_file": row.get("target_enroll_file", ""),
            "adv_file": row.get("adv_file", ""),
            "defended_file": "",
            "source_name": row.get("source_name", ""),
            "target_name": row.get("target_name", ""),
            "threshold": row.get("threshold", ""),
            "similarity_before": row.get("similarity_before", ""),
            "similarity_after_attack": row.get("similarity_after_attack", ""),
            "similarity_after_defense": "",
            "accepted_before": row.get("accepted_before", ""),
            "accepted_after_attack": row.get("accepted_after_attack", ""),
            "accepted_after_defense": "",
            "attack_success_before_defense": row.get("attack_success_before_defense", ""),
            "attack_success_after_defense": "",
            "defense_success": "",
            "epsilon": row.get("epsilon", ""),
            "alpha": row.get("alpha", ""),
            "steps": row.get("steps", ""),
            "l2": row.get("l2", ""),
            "linf": row.get("linf", ""),
            "defense_time_sec": "",
            "status": "pending",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Template rows: {len(output_rows)}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
