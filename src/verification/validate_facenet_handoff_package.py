"""Validate an extracted FaceNet verification attack handoff package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_FILES = [
    "attack_handoff_index.csv",
    "verification_metrics.json",
    "verification_attack_summary.csv",
    "manifest.json",
    "README.md",
]


REQUIRED_COLUMNS = {
    "sample_id",
    "source_file",
    "adv_file",
    "target_enroll_file",
    "perturbation_file",
    "threshold",
    "similarity_after_attack",
    "attack_success_before_defense",
    "epsilon",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate FaceNet handoff package contents.")
    parser.add_argument("--package-dir", type=Path, default=Path("outputs/handoff/facenet_verification_attack_package"))
    args = parser.parse_args()

    missing_top = [name for name in REQUIRED_FILES if not (args.package_dir / name).exists()]
    if missing_top:
        raise FileNotFoundError(f"Missing package files: {missing_top}")

    manifest = json.loads((args.package_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_missing = manifest.get("missing_files", [])
    if manifest_missing:
        raise FileNotFoundError(f"Manifest reports missing files: {manifest_missing[:10]}")

    index_path = args.package_dir / "attack_handoff_index.csv"
    with index_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("attack_handoff_index.csv has no header")
        missing_cols = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing_cols:
            raise ValueError(f"Missing required columns: {sorted(missing_cols)}")
        rows = list(reader)

    if not rows:
        raise ValueError("attack_handoff_index.csv has no rows")

    missing_refs: list[str] = []
    for row in rows:
        for column in ["adv_file", "target_enroll_file", "source_file", "perturbation_file"]:
            path = args.package_dir / row[column]
            if not path.exists():
                missing_refs.append(f"{row['sample_id']} {column}: {row[column]}")

    if missing_refs:
        raise FileNotFoundError(f"Missing referenced files: {missing_refs[:20]}")

    epsilons = sorted({row["epsilon"] for row in rows})
    print(f"Package: {args.package_dir}")
    print(f"Rows: {len(rows)}")
    print(f"Epsilons: {', '.join(epsilons)}")
    print("Validation: ok")


if __name__ == "__main__":
    main()
