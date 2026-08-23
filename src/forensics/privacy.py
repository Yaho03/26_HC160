"""Deterministic pseudonymization for publishable forensics artifacts."""

from __future__ import annotations

import hashlib
import re


def pseudonym(value: str, prefix: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    if re.fullmatch(rf"{re.escape(prefix)}_[0-9a-f]{{12}}", value):
        return value
    digest = hashlib.sha256(f"hc160:{prefix}:{value}".encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def artifact_reference(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    if re.fullmatch(r"artifact_[0-9a-f]{16}", value):
        return value
    digest = hashlib.sha256(f"hc160:artifact:{value}".encode("utf-8")).hexdigest()[:16]
    return f"artifact_{digest}"


def sanitize_identity_and_paths(row: dict[str, object]) -> dict[str, object]:
    sanitized = dict(row)
    identity_fields = {
        "source_identity": "source",
        "source_name": "source",
        "target_identity": "target",
        "target_name": "target",
        "account_id": "account",
    }
    for field, prefix in identity_fields.items():
        if field in sanitized:
            sanitized[field] = pseudonym(str(sanitized.get(field, "")), prefix)
    for field in list(sanitized):
        if field.endswith(("_file", "_path")):
            sanitized[field] = artifact_reference(str(sanitized.get(field, "")))
    return sanitized
