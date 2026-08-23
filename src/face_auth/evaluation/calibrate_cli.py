from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from src.face_auth.evaluation.calibration import calibrate_threshold


_SCORES = {
    "identity_similarity": True,
    "pad_live_score": True,
    "camera_motion_score": False,
    "content_replay_run": False,
    "adversarial_distance": False,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate face-auth thresholds on validation CSV"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--max-clean-frr", type=float, default=0.05)
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.input).open(encoding="utf-8")))
    labels = {row.get("label") for row in rows}
    if not {"clean", "attack"}.issubset(labels):
        raise ValueError("Validation CSV requires label=clean and label=attack rows")

    results = {}
    for column, higher_is_clean in _SCORES.items():
        clean = _values(rows, column, "clean")
        attack = _values(rows, column, "attack")
        if clean and attack:
            results[column] = calibrate_threshold(
                clean,
                attack,
                higher_is_clean=higher_is_clean,
                max_clean_frr=args.max_clean_frr,
            ).to_dict()

    payload = {
        "threshold_version": args.version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(Path(args.input)),
        "split": "validation",
        "max_clean_frr": args.max_clean_frr,
        "results": results,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _values(rows: list[dict[str, str]], column: str, label: str) -> list[float]:
    return [
        float(row[column])
        for row in rows
        if row.get("label") == label and row.get(column, "").strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
