from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from src.face_auth.inference.verifier import build_template


@dataclass(frozen=True)
class EnrollmentTemplate:
    embedding: np.ndarray
    template_version: str
    model_version: str
    alignment_version: str
    created_at: datetime


def create_template(
    crops: list[Image.Image],
    embedder,
    *,
    alignment_version: str = "mtcnn-crop-v1",
    min_frames: int = 5,
) -> EnrollmentTemplate:
    embeddings = embedder.embed(crops)
    aggregated = build_template(embeddings, min_frames=min_frames)
    return EnrollmentTemplate(
        embedding=aggregated,
        template_version=f"template-{uuid.uuid4()}",
        model_version=embedder.model_version,
        alignment_version=alignment_version,
        created_at=datetime.now(timezone.utc),
    )


def save_template(template: EnrollmentTemplate, path: str | Path) -> None:
    """Prototype storage only. Production storage must encrypt biometric templates."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "template_version": template.template_version,
        "model_version": template.model_version,
        "alignment_version": template.alignment_version,
        "created_at": template.created_at.isoformat(),
    }
    np.savez_compressed(
        target,
        embedding=template.embedding.astype(np.float32),
        metadata=np.array(json.dumps(metadata)),
    )


def load_template(path: str | Path) -> EnrollmentTemplate:
    with np.load(Path(path), allow_pickle=False) as payload:
        embedding = payload["embedding"].astype(np.float32)
        metadata = json.loads(str(payload["metadata"].item()))
    return EnrollmentTemplate(
        embedding=embedding,
        template_version=metadata["template_version"],
        model_version=metadata["model_version"],
        alignment_version=metadata["alignment_version"],
        created_at=datetime.fromisoformat(metadata["created_at"]),
    )
