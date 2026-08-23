"""Build a defense-team handoff package for verification attack outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


HANDOFF_COLUMNS = [
    "sample_id",
    "pair_id",
    "attack",
    "model",
    "pretrained",
    "source_file",
    "target_enroll_file",
    "adv_file",
    "perturbation_file",
    "source_name",
    "target_name",
    "threshold",
    "similarity_before",
    "similarity_after_attack",
    "similarity_gain",
    "accepted_before",
    "accepted_after_attack",
    "attack_success_before_defense",
    "epsilon",
    "alpha",
    "steps",
    "l2",
    "linf",
    "time_sec",
]


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: object) -> float:
    return float(str(value).strip())


def make_sample_id(row: dict[str, str]) -> str:
    key = "|".join([
        row.get("pair_id", ""),
        row.get("attack", ""),
        row.get("model", ""),
        row.get("pretrained", ""),
        row.get("epsilon", ""),
        row.get("alpha", ""),
        row.get("steps", ""),
        row.get("source_file", ""),
        row.get("target_enroll_file", ""),
    ])
    return "vf_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metadata_files(root: Path) -> list[Path]:
    return sorted(root.rglob("metadata_*.csv"))


def wanted_epsilon(row: dict[str, str], epsilons: set[str] | None) -> bool:
    if epsilons is None:
        return True
    epsilon = f"{parse_float(row.get('epsilon', 0)):.3f}"
    return epsilon in epsilons


def package_relpath(path: Path, package_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(package_root.resolve()))
    except ValueError:
        return path.name


def copy_if_exists(src: Path, dst: Path) -> str:
    if not src.exists():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package FaceNet verification attack outputs for defense handoff.")
    parser.add_argument("--metadata-root", type=Path, default=Path("outputs/verification_attacks_facenet"))
    parser.add_argument("--verification-metrics", type=Path, default=Path("outputs/verification_facenet/verification_metrics.json"))
    parser.add_argument("--attack-summary", type=Path, default=Path("outputs/verification_attacks_facenet/verification_attack_summary.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/handoff/facenet_verification_attack_package"))
    parser.add_argument("--zip-out", type=Path, default=Path("outputs/handoff/facenet_verification_attack_package.zip"))
    parser.add_argument(
        "--epsilons",
        default="0.005,0.010",
        help="Comma-separated epsilons to include, formatted like 0.005,0.010. Use ALL for every metadata row.",
    )
    parser.add_argument("--successful-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--copy-source-and-target", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    epsilons = None
    if args.epsilons.strip().upper() != "ALL":
        epsilons = {f"{float(item):.3f}" for item in args.epsilons.split(",") if item.strip()}

    files = metadata_files(args.metadata_root)
    if not files:
        raise FileNotFoundError(f"No metadata files found under {args.metadata_root}")

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    missing_files: list[str] = []

    for path in files:
        for row in read_metadata(path):
            if not wanted_epsilon(row, epsilons):
                continue
            if args.successful_only and not parse_bool(row.get("attack_success", False)):
                continue

            sample_id = make_sample_id(row)
            epsilon = f"{parse_float(row.get('epsilon', 0)):.3f}"
            sample_dir = args.out_dir / "samples" / epsilon / sample_id

            src_adv = Path(row["adv_file"])
            src_perturb = Path(row["perturbation_file"])
            src_source = Path(row["source_file"])
            src_target = Path(row["target_enroll_file"])

            adv_dst = sample_dir / "adversarial" / src_adv.name
            perturb_dst = sample_dir / "perturbation" / src_perturb.name
            source_dst = sample_dir / "source" / src_source.name
            target_dst = sample_dir / "target_enroll" / src_target.name

            adv_copied = copy_if_exists(src_adv, adv_dst)
            perturb_copied = copy_if_exists(src_perturb, perturb_dst)
            source_copied = copy_if_exists(src_source, source_dst) if args.copy_source_and_target else ""
            target_copied = copy_if_exists(src_target, target_dst) if args.copy_source_and_target else ""

            for label, copied, original in [
                ("adv_file", adv_copied, src_adv),
                ("perturbation_file", perturb_copied, src_perturb),
                ("source_file", source_copied, src_source),
                ("target_enroll_file", target_copied, src_target),
            ]:
                if args.copy_source_and_target or label in {"adv_file", "perturbation_file"}:
                    if not copied:
                        missing_files.append(f"{label}: {original}")

            rows.append({
                "sample_id": sample_id,
                "pair_id": row.get("pair_id", ""),
                "attack": row.get("attack", ""),
                "model": row.get("model", ""),
                "pretrained": row.get("pretrained", ""),
                "source_file": package_relpath(Path(source_copied), args.out_dir) if source_copied else row.get("source_file", ""),
                "target_enroll_file": package_relpath(Path(target_copied), args.out_dir) if target_copied else row.get("target_enroll_file", ""),
                "adv_file": package_relpath(Path(adv_copied), args.out_dir) if adv_copied else row.get("adv_file", ""),
                "perturbation_file": package_relpath(Path(perturb_copied), args.out_dir) if perturb_copied else row.get("perturbation_file", ""),
                "source_name": row.get("source_name", ""),
                "target_name": row.get("target_name", ""),
                "threshold": row.get("threshold", ""),
                "similarity_before": row.get("similarity_before", ""),
                "similarity_after_attack": row.get("similarity_after", ""),
                "similarity_gain": row.get("similarity_gain", ""),
                "accepted_before": row.get("accepted_before", ""),
                "accepted_after_attack": row.get("accepted_after", ""),
                "attack_success_before_defense": row.get("attack_success", ""),
                "epsilon": row.get("epsilon", ""),
                "alpha": row.get("alpha", ""),
                "steps": row.get("steps", ""),
                "l2": row.get("l2", ""),
                "linf": row.get("linf", ""),
                "time_sec": row.get("time_sec", ""),
            })

    if not rows:
        raise ValueError("No rows selected for handoff package.")

    index_path = args.out_dir / "attack_handoff_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HANDOFF_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    if args.verification_metrics.exists():
        shutil.copy2(args.verification_metrics, args.out_dir / "verification_metrics.json")
    if args.attack_summary.exists():
        shutil.copy2(args.attack_summary, args.out_dir / "verification_attack_summary.csv")

    manifest = {
        "package": "facenet_verification_attack_package",
        "selected_epsilons": sorted(epsilons) if epsilons is not None else "ALL",
        "successful_only": args.successful_only,
        "samples": len(rows),
        "metadata_files": [str(path) for path in files],
        "index": "attack_handoff_index.csv",
        "verification_metrics": "verification_metrics.json",
        "attack_summary": "verification_attack_summary.csv",
        "missing_files": missing_files,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    readme = f"""# FaceNet Verification Attack Handoff Package

This package contains targeted PGD impersonation attack outputs for defense evaluation.

## How to use

1. Read `attack_handoff_index.csv`.
2. For each row, use `adv_file` as the defense input image.
3. Save the defended image path as `defended_file`.
4. Recompute FaceNet cosine similarity between `defended_file` and `target_enroll_file`.
5. Use the threshold in the row, or `verification_metrics.json`, to decide whether the attack still succeeds.

## Defense success definition

```text
accepted_after_attack == True
accepted_after_defense == False
```

## Package summary

- selected epsilons: {manifest["selected_epsilons"]}
- successful only: {args.successful_only}
- samples: {len(rows)}
- missing files: {len(missing_files)}

"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")

    args.zip_out.parent.mkdir(parents=True, exist_ok=True)
    if args.zip_out.exists():
        args.zip_out.unlink()
    with zipfile.ZipFile(args.zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(args.out_dir.rglob("*")):
            if file.is_file():
                zf.write(file, arcname=str(file.relative_to(args.out_dir.parent)))

    print(f"Metadata files: {len(files)}")
    print(f"Selected rows: {len(rows)}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Index: {index_path}")
    print(f"Package dir: {args.out_dir}")
    print(f"Zip: {args.zip_out}")


if __name__ == "__main__":
    main()
