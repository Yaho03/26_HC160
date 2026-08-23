"""Small reproducibility utilities with optional NumPy/PyTorch support."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def set_global_seed(seed: int, deterministic: bool = False) -> dict[str, bool]:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    status = {"python": True, "numpy": False, "torch": False, "deterministic": False}

    try:
        import numpy as np

        np.random.seed(seed)
        status["numpy"] = True
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            status["deterministic"] = True
        status["torch"] = True
    except ImportError:
        pass
    return status


def git_state(repo_dir: str | Path = ".") -> dict[str, str | bool]:
    root = Path(repo_dir)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"git_commit": commit, "dirty_worktree": dirty}
