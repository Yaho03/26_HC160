"""Export provenance-bound FaceNet cosine scores for EXP-VER-001."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from src.common.reproducibility import git_state, sha256_file, stable_json_bytes
from src.datasets.manifest import (
    DatasetManifestError,
    DatasetManifestRow,
    load_manifest,
    validate_manifest_rows,
)
from src.evaluation.verification_calibration import VerificationScore


class ScoreExportError(ValueError):
    """Raised when score export inputs violate the frozen protocol."""


@dataclass(frozen=True)
class VerificationPair:
    schema_version: str
    pair_id: str
    protocol_id: str
    left_sample_id: str
    right_sample_id: str
    same_identity: bool
    split: str
    fold: int | None = None
    pair_group_id: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--pair-manifest-id", required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--model-artifact-id", required=True)
    parser.add_argument("--preprocessing-config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-identity-overlap", action="store_true")
    parser.add_argument("--repo-dir", type=Path, default=Path("."))
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--created-at")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        _validate_output_paths(
            (args.scores_output, args.metadata_output), overwrite=args.overwrite
        )
        code_state = _read_code_state(args.repo_dir)
        if code_state["dirty_worktree"] and not args.allow_dirty:
            raise ScoreExportError(
                "refusing an export from a dirty worktree; commit changes or pass --allow-dirty"
            )

        input_hashes = {
            "dataset_manifest": sha256_file(args.dataset_manifest),
            "pair_manifest": sha256_file(args.pair_manifest),
            "model_checkpoint": sha256_file(args.model_checkpoint),
            "preprocessing_config": sha256_file(args.preprocessing_config),
        }
        preprocessing = load_preprocessing_config(args.preprocessing_config)
        dataset_rows = load_manifest(args.dataset_manifest)
        dataset_report = validate_manifest_rows(
            dataset_rows,
            artifact_root=args.artifact_root,
            require_identity_disjoint=not args.allow_identity_overlap,
        )
        pairs = load_pair_manifest(args.pair_manifest)
        split = validate_pair_protocol(
            pairs,
            protocol_id=args.protocol_id,
            dataset_rows=dataset_rows,
        )
        embedder = FaceNetVGGFace2Embedder(
            checkpoint=args.model_checkpoint,
            preprocessing=preprocessing,
            device=args.device,
            batch_size=args.batch_size,
        )
        scores = score_verification_pairs(
            pairs,
            dataset_rows,
            artifact_root=args.artifact_root,
            embed_many=embedder.embed_many,
            model_artifact_id=args.model_artifact_id,
            preprocessing_artifact_id=preprocessing["preprocessing_artifact_id"],
        )
        score_payload = b"".join(
            stable_json_bytes(asdict(record)) + b"\n" for record in scores
        )

        _verify_inputs_unchanged(args, input_hashes, dataset_rows)
        if _read_code_state(args.repo_dir) != code_state:
            raise ScoreExportError("repository state changed during score export")

        created_at = args.created_at or datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        score_sha256 = hashlib.sha256(score_payload).hexdigest()
        metadata = {
            "schema_version": "1.0",
            "export_id": f"score_export_{score_sha256[:16]}",
            "run_id": args.run_id,
            "created_at": created_at,
            "code": code_state,
            "protocol_id": args.protocol_id,
            "split": split,
            "software": _software_versions(),
            "model_artifact": {
                "id": args.model_artifact_id,
                "sha256": input_hashes["model_checkpoint"],
                "bytes": args.model_checkpoint.stat().st_size,
                "architecture": "facenet_pytorch.InceptionResnetV1",
                "weights_source": "vggface2",
            },
            "preprocessing_artifact": {
                "id": preprocessing["preprocessing_artifact_id"],
                "sha256": input_hashes["preprocessing_config"],
                "bytes": args.preprocessing_config.stat().st_size,
            },
            "dataset_manifest": {
                "id": dataset_report.dataset_id,
                "sha256": input_hashes["dataset_manifest"],
                "row_count": dataset_report.row_count,
            },
            "pair_manifest": {
                "id": args.pair_manifest_id,
                "sha256": input_hashes["pair_manifest"],
                "row_count": len(pairs),
                "split": split,
            },
            "score_file": {
                "sha256": score_sha256,
                "bytes": len(score_payload),
                "row_count": len(scores),
            },
            "evaluation_config": {
                "device": args.device,
                "batch_size": args.batch_size,
                "require_identity_disjoint": not args.allow_identity_overlap,
                "deterministic_algorithms": True,
            },
        }
        _atomic_write_many(
            (
                (args.scores_output, score_payload),
                (args.metadata_output, stable_json_bytes(metadata) + b"\n"),
            ),
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "export_id": metadata["export_id"],
                    "split": split,
                    "pair_count": len(scores),
                    "unique_sample_count": len(
                        {pair.left_sample_id for pair in pairs}
                        | {pair.right_sample_id for pair in pairs}
                    ),
                },
                sort_keys=True,
            )
        )
        return 0
    except (
        DatasetManifestError,
        OSError,
        RuntimeError,
        ScoreExportError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
        return 2


def load_pair_manifest(path: str | Path) -> tuple[VerificationPair, ...]:
    required = {
        "schema_version",
        "pair_id",
        "protocol_id",
        "left_sample_id",
        "right_sample_id",
        "same_identity",
        "split",
    }
    allowed = required | {"fold", "pair_group_id"}
    pairs: list[VerificationPair] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScoreExportError(
                    f"invalid pair JSON on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise ScoreExportError(f"pair line {line_number} must be an object")
            missing = sorted(required - set(value))
            unknown = sorted(set(value) - allowed)
            if missing or unknown:
                details = []
                if missing:
                    details.append(f"missing={','.join(missing)}")
                if unknown:
                    details.append(f"unknown={','.join(unknown)}")
                raise ScoreExportError(
                    f"invalid pair fields on line {line_number}: {'; '.join(details)}"
                )
            pairs.append(VerificationPair(**value))
    if not pairs:
        raise ScoreExportError("pair manifest must contain at least one row")
    return tuple(pairs)


def validate_pair_protocol(
    pairs: Sequence[VerificationPair],
    *,
    protocol_id: str,
    dataset_rows: Sequence[DatasetManifestRow],
) -> str:
    sample_index = {row.sample_id: row for row in dataset_rows}
    pair_ids: set[str] = set()
    splits: set[str] = set()
    for pair in pairs:
        if pair.schema_version != "1.0":
            raise ScoreExportError("pair schema_version must be 1.0")
        if not isinstance(pair.pair_id, str) or not pair.pair_id.strip():
            raise ScoreExportError("pair_id must be a non-empty string")
        if pair.pair_id in pair_ids:
            raise ScoreExportError(f"duplicate or empty pair_id: {pair.pair_id}")
        pair_ids.add(pair.pair_id)
        for field in ("protocol_id", "left_sample_id", "right_sample_id", "split"):
            value = getattr(pair, field)
            if not isinstance(value, str) or not value.strip():
                raise ScoreExportError(
                    f"{field} must be a non-empty string: {pair.pair_id}"
                )
        if type(pair.same_identity) is not bool:
            raise ScoreExportError(f"same_identity must be boolean: {pair.pair_id}")
        if pair.fold is not None and (type(pair.fold) is not int or pair.fold < 0):
            raise ScoreExportError(f"fold must be a non-negative integer: {pair.pair_id}")
        if pair.pair_group_id is not None and not isinstance(pair.pair_group_id, str):
            raise ScoreExportError(f"pair_group_id must be a string: {pair.pair_id}")
        if pair.protocol_id != protocol_id:
            raise ScoreExportError(
                f"pair protocol differs from --protocol-id: {pair.pair_id}"
            )
        if pair.split not in {"calibration", "test"}:
            raise ScoreExportError(
                f"score export supports calibration/test only: {pair.pair_id}"
            )
        splits.add(pair.split)
        if pair.left_sample_id == pair.right_sample_id:
            raise ScoreExportError(f"pair repeats one sample: {pair.pair_id}")
        try:
            left = sample_index[pair.left_sample_id]
            right = sample_index[pair.right_sample_id]
        except KeyError as exc:
            raise ScoreExportError(
                f"pair references an unknown sample: {pair.pair_id}"
            ) from exc
        if left.split != pair.split or right.split != pair.split:
            raise ScoreExportError(
                f"pair/sample split mismatch: {pair.pair_id}"
            )
        if (left.identity_token == right.identity_token) is not pair.same_identity:
            raise ScoreExportError(
                f"pair label contradicts dataset identities: {pair.pair_id}"
            )
    if len(splits) != 1:
        raise ScoreExportError("one score export must contain exactly one split")
    return next(iter(splits))


def score_verification_pairs(
    pairs: Sequence[VerificationPair],
    dataset_rows: Sequence[DatasetManifestRow],
    *,
    artifact_root: str | Path,
    embed_many: Callable[[Sequence[Path]], Sequence[np.ndarray]],
    model_artifact_id: str,
    preprocessing_artifact_id: str,
) -> tuple[VerificationScore, ...]:
    index = {row.sample_id: row for row in dataset_rows}
    sample_ids = sorted(
        {pair.left_sample_id for pair in pairs}
        | {pair.right_sample_id for pair in pairs}
    )
    root = Path(artifact_root)
    embeddings = embed_many(
        [root / index[sample_id].relative_uri for sample_id in sample_ids]
    )
    if len(embeddings) != len(sample_ids):
        raise ScoreExportError("embedder returned an unexpected number of vectors")
    normalized: dict[str, np.ndarray] = {}
    dimension: int | None = None
    for sample_id, embedding in zip(sample_ids, embeddings):
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if not vector.size or not np.all(np.isfinite(vector)):
            raise ScoreExportError(f"invalid embedding for sample: {sample_id}")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            raise ScoreExportError(f"zero or invalid embedding for sample: {sample_id}")
        if dimension is None:
            dimension = vector.size
        elif vector.size != dimension:
            raise ScoreExportError("embedding dimensions are inconsistent")
        normalized[sample_id] = vector / norm

    return tuple(
        VerificationScore(
            pair_id=pair.pair_id,
            same_identity=pair.same_identity,
            score=max(
                -1.0,
                min(
                    1.0,
                    float(
                        np.dot(
                            normalized[pair.left_sample_id],
                            normalized[pair.right_sample_id],
                        )
                    ),
                ),
            ),
            split=pair.split,
            protocol_id=pair.protocol_id,
            model_artifact_id=model_artifact_id,
            preprocessing_artifact_id=preprocessing_artifact_id,
        )
        for pair in pairs
    )


def load_preprocessing_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version": "1.0",
        "implementation": "facenet_pytorch.InceptionResnetV1",
        "image_size": [160, 160],
        "color_space": "RGB",
        "resize": "bilinear",
        "pixel_subtract": 127.5,
        "pixel_divide": 128.0,
        "tensor_layout": "NCHW",
        "l2_normalize_embedding": True,
        "score_function": "cosine_similarity",
    }
    if not isinstance(value, dict):
        raise ScoreExportError("preprocessing config must be an object")
    if not isinstance(value.get("preprocessing_artifact_id"), str) or not value[
        "preprocessing_artifact_id"
    ].strip():
        raise ScoreExportError("preprocessing_artifact_id must be non-empty")
    comparable = dict(value)
    comparable.pop("preprocessing_artifact_id", None)
    if comparable != expected:
        raise ScoreExportError("unsupported FaceNet preprocessing contract")
    return value


class FaceNetVGGFace2Embedder:
    def __init__(
        self,
        *,
        checkpoint: Path,
        preprocessing: Mapping[str, Any],
        device: str,
        batch_size: int,
    ) -> None:
        import torch
        import torch.nn.functional as functional
        from facenet_pytorch import InceptionResnetV1

        self.torch = torch
        self.functional = functional
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.preprocessing = preprocessing
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
        self.model = InceptionResnetV1(
            pretrained=None,
            classify=True,
            num_classes=8631,
        ).to(self.device)
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.classify = False
        self.model.eval()

    def embed_many(self, paths: Sequence[Path]) -> list[np.ndarray]:
        embeddings: list[np.ndarray] = []
        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start : start + self.batch_size]
            tensors = [self._tensor(path) for path in batch_paths]
            batch = self.torch.stack(tensors).to(self.device)
            with self.torch.no_grad():
                output = self.model(batch)
                output = self.functional.normalize(output, p=2, dim=1)
            embeddings.extend(output.detach().cpu().numpy())
        return embeddings

    def _tensor(self, path: Path):
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        image_size = tuple(self.preprocessing["image_size"])
        with Image.open(path) as image:
            resized = image.convert("RGB").resize(image_size, resampling)
            array = np.asarray(resized, dtype=np.float32)
        array = (
            array - float(self.preprocessing["pixel_subtract"])
        ) / float(self.preprocessing["pixel_divide"])
        return self.torch.from_numpy(array.transpose(2, 0, 1)).float()


def _software_versions() -> dict[str, str]:
    import torch

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "facenet_pytorch": version("facenet-pytorch"),
        "numpy": np.__version__,
        "pillow": version("Pillow"),
    }


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("pair_manifest_id", "protocol_id", "model_artifact_id", "run_id"):
        if not getattr(args, name).strip():
            raise ScoreExportError(f"--{name.replace('_', '-')} must not be empty")
    if args.batch_size <= 0:
        raise ScoreExportError("--batch-size must be positive")
    if args.created_at is not None:
        try:
            datetime.fromisoformat(args.created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ScoreExportError("--created-at must be an ISO 8601 timestamp") from exc
    resolved_inputs = {
        path.resolve()
        for path in (
            args.dataset_manifest,
            args.pair_manifest,
            args.model_checkpoint,
            args.preprocessing_config,
        )
    }
    if (
        args.scores_output.resolve() in resolved_inputs
        or args.metadata_output.resolve() in resolved_inputs
    ):
        raise ScoreExportError("outputs must differ from all input files")


def _validate_output_paths(paths: tuple[Path, ...], *, overwrite: bool) -> None:
    if len({path.resolve() for path in paths}) != len(paths):
        raise ScoreExportError("score and metadata outputs must be different files")
    if not overwrite:
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise ScoreExportError(
                f"refusing to overwrite existing file(s): {', '.join(existing)}"
            )


def _read_code_state(repo_dir: Path) -> dict[str, str | bool]:
    try:
        return git_state(repo_dir)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ScoreExportError(f"unable to identify repository state: {exc}") from exc


def _verify_inputs_unchanged(
    args: argparse.Namespace,
    expected: Mapping[str, str],
    dataset_rows: Sequence[DatasetManifestRow],
) -> None:
    paths = {
        "dataset_manifest": args.dataset_manifest,
        "pair_manifest": args.pair_manifest,
        "model_checkpoint": args.model_checkpoint,
        "preprocessing_config": args.preprocessing_config,
    }
    for name, path in paths.items():
        if sha256_file(path) != expected[name]:
            raise ScoreExportError(f"{name.replace('_', ' ')} changed during export")
    validate_manifest_rows(
        dataset_rows,
        artifact_root=args.artifact_root,
        require_identity_disjoint=not args.allow_identity_overlap,
    )


def _atomic_write_many(
    outputs: tuple[tuple[Path, bytes], ...], *, overwrite: bool
) -> None:
    _validate_output_paths(tuple(path for path, _ in outputs), overwrite=overwrite)
    temporary_files: list[tuple[Path, Path]] = []
    try:
        for path, payload in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
            temporary_files.append((temporary, path))
        committed: list[Path] = []
        try:
            for temporary, path in temporary_files:
                if overwrite:
                    os.replace(temporary, path)
                else:
                    os.link(temporary, path)
                    temporary.unlink()
                committed.append(path)
        except BaseException:
            if not overwrite:
                for path in committed:
                    path.unlink(missing_ok=True)
            raise
    finally:
        for temporary, _ in temporary_files:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
