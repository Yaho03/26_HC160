"""Summarize targeted verification attack metadata files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_file(path: Path) -> dict[str, object]:
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"No rows in {path}")

    first = rows[0]
    attack_successes = [parse_bool(row["attack_success"]) for row in rows]
    success_from_rejects = [parse_bool(row["success_from_reject"]) for row in rows]
    accepted_before = [parse_bool(row["accepted_before"]) for row in rows]
    accepted_after = [parse_bool(row["accepted_after"]) for row in rows]

    return {
        "metadata_file": str(path),
        "attack": first.get("attack", ""),
        "epsilon": first.get("epsilon", ""),
        "alpha": first.get("alpha", ""),
        "steps": first.get("steps", ""),
        "only_initial_rejects": first.get("only_initial_rejects", ""),
        "samples": len(rows),
        "accepted_before_rate": mean([float(v) for v in accepted_before]),
        "accepted_after_rate": mean([float(v) for v in accepted_after]),
        "target_accept_rate_after_attack": mean([float(v) for v in attack_successes]),
        "success_from_reject_rate": mean([float(v) for v in success_from_rejects]),
        "avg_similarity_before": mean([float(row["similarity_before"]) for row in rows]),
        "avg_similarity_after": mean([float(row["similarity_after"]) for row in rows]),
        "avg_similarity_gain": mean([float(row["similarity_gain"]) for row in rows]),
        "avg_l2": mean([float(row["l2"]) for row in rows]),
        "avg_linf": mean([float(row["linf"]) for row in rows]),
        "avg_time_sec": mean([float(row["time_sec"]) for row in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize verification attack metadata CSVs.")
    parser.add_argument("--metadata-root", type=Path, default=Path("outputs/verification_attacks"))
    parser.add_argument("--out", type=Path, default=Path("outputs/verification_attacks/verification_attack_summary.csv"))
    args = parser.parse_args()

    files = sorted(args.metadata_root.rglob("metadata_*.csv"))
    if not files:
        raise FileNotFoundError(f"No metadata CSV files found under {args.metadata_root}")

    rows = [summarize_file(path) for path in files]
    rows.sort(key=lambda row: (str(row["attack"]), float(row["epsilon"] or 0), str(row["metadata_file"])))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Metadata files: {len(files)}")
    print(f"Saved: {args.out}")
    print("=== verification attack summary ===")
    with args.out.open(encoding="utf-8") as f:
        print(f.read().strip())


if __name__ == "__main__":
    main()
