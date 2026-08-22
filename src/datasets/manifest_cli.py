"""Command-line interface for EXP-DATA-001 dataset manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.reproducibility import sha256_file
from src.datasets.manifest import (
    DatasetManifestError,
    build_snapshot_metadata,
    discover_manifest_rows,
    load_manifest,
    validate_manifest_rows,
    write_manifest,
    write_snapshot_metadata,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build manifest and snapshot metadata")
    build.add_argument("--artifact-root", type=Path, required=True)
    build.add_argument("--manifest-output", type=Path, required=True)
    build.add_argument("--metadata-output", type=Path, required=True)
    build.add_argument("--manifest-uri")
    build.add_argument("--dataset-id", required=True)
    build.add_argument("--license-id", required=True)
    build.add_argument("--source-uri", required=True)
    build.add_argument("--source-retrieved-at", required=True)
    archive = build.add_mutually_exclusive_group(required=True)
    archive.add_argument("--source-archive", type=Path)
    archive.add_argument("--source-archive-sha256")
    build.add_argument("--require-identity-disjoint", action="store_true")
    build.add_argument("--overwrite", action="store_true")

    validate = subparsers.add_parser("validate", help="validate an existing manifest")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--artifact-root", type=Path)
    validate.add_argument("--require-identity-disjoint", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            if not args.overwrite:
                existing = [
                    str(path)
                    for path in (args.manifest_output, args.metadata_output)
                    if path.exists()
                ]
                if existing:
                    raise DatasetManifestError(
                        f"refusing to overwrite existing file(s): {', '.join(existing)}"
                    )
            rows = discover_manifest_rows(
                args.artifact_root,
                dataset_id=args.dataset_id,
                license_id=args.license_id,
                source_uri=args.source_uri,
            )
            report = validate_manifest_rows(
                rows,
                artifact_root=args.artifact_root,
                require_identity_disjoint=args.require_identity_disjoint,
            )
            archive_hash = (
                sha256_file(args.source_archive)
                if args.source_archive is not None
                else args.source_archive_sha256
            )
            metadata = build_snapshot_metadata(
                dataset_id=args.dataset_id,
                manifest_relative_uri=args.manifest_uri or args.manifest_output.name,
                report=report,
                source_archive_sha256=archive_hash,
                source_uri=args.source_uri,
                source_retrieved_at=args.source_retrieved_at,
                license_id=args.license_id,
            )
            manifest_hash = write_manifest(rows, args.manifest_output, args.overwrite)
            if manifest_hash != report.manifest_sha256:
                raise DatasetManifestError("written manifest hash differs from validation hash")
            write_snapshot_metadata(metadata, args.metadata_output, args.overwrite)
            print(json.dumps({"status": "ok", **report.to_dict()}, sort_keys=True))
            return 0

        rows = load_manifest(args.manifest)
        report = validate_manifest_rows(
            rows,
            artifact_root=args.artifact_root,
            require_identity_disjoint=args.require_identity_disjoint,
        )
        print(json.dumps({"status": "ok", **report.to_dict()}, sort_keys=True))
        return 0
    except (DatasetManifestError, OSError) as exc:
        parser = build_parser()
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
