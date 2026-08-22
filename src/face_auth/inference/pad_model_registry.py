from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from src.common.reproducibility import sha256_file


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PADModelArtifact:
    schema_version: str
    model_id: str
    runtime: str
    validation_status: str
    filename: str
    source_url: str
    source_page: str
    sha256: str
    byte_count: int
    license_name: str
    license_url: str
    input_name: str
    input_shape: tuple[int, ...]
    color_order: str
    mean: tuple[float, float, float]
    scale: tuple[float, float, float]
    output_name: str
    output_kind: str
    output_shape: tuple[int, ...]
    live_class_index: int
    classes: dict[str, str]


@dataclass(frozen=True)
class PADModelVerification:
    model_id: str
    runtime: str
    validation_status: str
    sha256: str
    byte_count: int
    verified: bool = True

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "runtime": self.runtime,
            "validation_status": self.validation_status,
            "sha256": self.sha256,
            "bytes": self.byte_count,
            "verified": self.verified,
        }


def load_pad_model_artifact(path: str | Path) -> PADModelArtifact:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to load PAD model registry: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("PAD model registry must be a JSON object")
    _exact_keys(
        payload,
        {
            "schema_version",
            "model_id",
            "task",
            "runtime",
            "validation_status",
            "artifact",
            "input",
            "output",
        },
        "registry",
    )
    artifact = _mapping(payload["artifact"], "artifact")
    model_input = _mapping(payload["input"], "input")
    output = _mapping(payload["output"], "output")
    _exact_keys(
        artifact,
        {
            "filename",
            "source_url",
            "source_page",
            "sha256",
            "bytes",
            "license",
            "license_url",
        },
        "artifact",
    )
    _exact_keys(
        model_input,
        {"name", "layout", "shape", "color_order", "mean", "scale"},
        "input",
    )
    _exact_keys(
        output,
        {"name", "kind", "shape", "live_class_index", "classes"},
        "output",
    )
    model = PADModelArtifact(
        schema_version=str(payload["schema_version"]),
        model_id=str(payload["model_id"]),
        runtime=str(payload["runtime"]),
        validation_status=str(payload["validation_status"]),
        filename=str(artifact["filename"]),
        source_url=str(artifact["source_url"]),
        source_page=str(artifact["source_page"]),
        sha256=str(artifact["sha256"]),
        byte_count=_positive_int(artifact["bytes"], "artifact.bytes"),
        license_name=str(artifact["license"]),
        license_url=str(artifact["license_url"]),
        input_name=str(model_input["name"]),
        input_shape=_shape(model_input["shape"], "input.shape"),
        color_order=str(model_input["color_order"]),
        mean=_triplet(model_input["mean"], "input.mean"),
        scale=_triplet(model_input["scale"], "input.scale"),
        output_name=str(output["name"]),
        output_kind=str(output["kind"]),
        output_shape=_shape(output["shape"], "output.shape"),
        live_class_index=_non_negative_int(
            output["live_class_index"], "output.live_class_index"
        ),
        classes={
            str(key): str(value)
            for key, value in _mapping(output["classes"], "output.classes").items()
        },
    )
    _validate_model(model, task=str(payload["task"]), layout=str(model_input["layout"]))
    return model


def verify_pad_model_artifact(
    model: PADModelArtifact, artifact_path: str | Path
) -> PADModelVerification:
    path = Path(artifact_path)
    if not path.is_file():
        raise ValueError(f"PAD model artifact does not exist: {path}")
    actual_size = path.stat().st_size
    if actual_size != model.byte_count:
        raise ValueError(
            f"PAD model byte count mismatch: expected {model.byte_count}, got {actual_size}"
        )
    actual_hash = sha256_file(path)
    if actual_hash != model.sha256:
        raise ValueError(
            f"PAD model SHA-256 mismatch: expected {model.sha256}, got {actual_hash}"
        )
    return PADModelVerification(
        model_id=model.model_id,
        runtime=model.runtime,
        validation_status=model.validation_status,
        sha256=actual_hash,
        byte_count=actual_size,
    )


def _validate_model(model: PADModelArtifact, *, task: str, layout: str) -> None:
    if model.schema_version != "1.0":
        raise ValueError(
            f"Unsupported PAD model registry version: {model.schema_version}"
        )
    if task != "presentation_attack_detection":
        raise ValueError(f"Unsupported PAD model task: {task}")
    if model.runtime != "onnx":
        raise ValueError(f"Unsupported PAD model runtime: {model.runtime}")
    if model.validation_status not in {"candidate_unvalidated", "validated"}:
        raise ValueError("Unsupported PAD model validation_status")
    if not _SHA256.fullmatch(model.sha256):
        raise ValueError("artifact.sha256 must be 64 lowercase hexadecimal characters")
    if not model.filename.endswith(".onnx"):
        raise ValueError("ONNX PAD artifact filename must end with .onnx")
    for name, url in {
        "artifact.source_url": model.source_url,
        "artifact.source_page": model.source_page,
        "artifact.license_url": model.license_url,
    }.items():
        if not url.startswith("https://"):
            raise ValueError(f"{name} must use HTTPS")
    if layout != "NCHW" or model.input_shape != (1, 3, 128, 128):
        raise ValueError("Registered anti-spoof model input must be NCHW [1,3,128,128]")
    if model.color_order != "RGB":
        raise ValueError("Registered anti-spoof ONNX input must use RGB")
    if any(value == 0 for value in model.scale):
        raise ValueError("input.scale values must be non-zero")
    if not all(math.isfinite(value) for value in (*model.mean, *model.scale)):
        raise ValueError("Registered anti-spoof preprocessing values must be finite")
    if model.output_kind != "probability" or model.output_shape != (1, 2):
        raise ValueError("Registered anti-spoof output must be two probabilities")
    if model.classes.get(str(model.live_class_index)) != "bona_fide":
        raise ValueError("live_class_index must identify the bona_fide class")
    if model.classes != {"0": "bona_fide", "1": "spoof"}:
        raise ValueError("Registered anti-spoof classes must be bona_fide and spoof")


def _exact_keys(value: dict, expected: set[str], name: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(f"Invalid {name} keys: missing={missing}, unknown={unknown}")


def _mapping(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _shape(value, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty integer array")
    return tuple(_positive_int(item, name) for item in value)


def _triplet(value, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain three numbers")
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain three numbers") from error


def _positive_int(value, name: str) -> int:
    result = _non_negative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
