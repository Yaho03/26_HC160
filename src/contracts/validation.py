"""Dependency-free semantic checks complementing the JSON Schemas."""

from __future__ import annotations

import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Mapping


class ContractError(ValueError):
    pass


_HEX_64 = re.compile(r"^[a-f0-9]{64}$")


def validate_relative_uri(value: str) -> str:
    if not value or not value.strip():
        raise ContractError("relative_uri must not be empty")
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ContractError(f"absolute artifact path is forbidden: {value}")
    if ".." in PurePosixPath(value).parts or ".." in PureWindowsPath(value).parts:
        raise ContractError(f"parent traversal is forbidden: {value}")
    return value


def validate_sha256(value: str) -> str:
    if not _HEX_64.fullmatch(value):
        raise ContractError("sha256 must be 64 lowercase hexadecimal characters")
    return value


def validate_verification_pair(record: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "pair_id",
        "protocol_id",
        "left_sample_id",
        "right_sample_id",
        "same_identity",
        "split",
    }
    _require(record, required)
    if record["left_sample_id"] == record["right_sample_id"]:
        raise ContractError("verification pair cannot reference the same sample twice")
    if type(record["same_identity"]) is not bool:
        raise ContractError("same_identity must be boolean")


def validate_attack_result(record: Mapping[str, Any]) -> None:
    required = {
        "accepted_before",
        "accepted_after",
        "success_from_reject",
        "elapsed_ms",
    }
    _require(record, required)
    before = _bool(record["accepted_before"], "accepted_before")
    after = _bool(record["accepted_after"], "accepted_after")
    success = _bool(record["success_from_reject"], "success_from_reject")
    if success != ((not before) and after):
        raise ContractError("success_from_reject must equal !accepted_before && accepted_after")
    _non_negative(record["elapsed_ms"], "elapsed_ms")
    for field in ("epsilon", "alpha", "steps", "queries_used", "l0", "l2", "linf"):
        if field in record and record[field] is not None:
            _non_negative(record[field], field)
    if "query_budget" in record and record.get("queries_used") is not None:
        if record["queries_used"] > record["query_budget"]:
            raise ContractError("queries_used exceeds query_budget")


def validate_defense_result(record: Mapping[str, Any]) -> None:
    required = {
        "input_kind",
        "attack_result_id",
        "clean_pair_id",
        "accepted_before",
        "accepted_after",
        "defense_success",
        "elapsed_ms",
    }
    _require(record, required)
    before = _bool(record["accepted_before"], "accepted_before")
    after = _bool(record["accepted_after"], "accepted_after")
    _non_negative(record["elapsed_ms"], "elapsed_ms")

    if record["input_kind"] == "attack":
        if not record["attack_result_id"] or record["clean_pair_id"] is not None:
            raise ContractError("attack input requires only attack_result_id")
        success = _bool(record["defense_success"], "defense_success")
        if success != (before and not after):
            raise ContractError("defense_success must equal accepted_before && !accepted_after")
    elif record["input_kind"] == "clean":
        if not record["clean_pair_id"] or record["attack_result_id"] is not None:
            raise ContractError("clean input requires only clean_pair_id")
        if record["defense_success"] is not None:
            raise ContractError("clean input must not define defense_success")
    else:
        raise ContractError(f"unsupported input_kind: {record['input_kind']}")


def _require(record: Mapping[str, Any], fields: set[str]) -> None:
    missing = sorted(field for field in fields if field not in record)
    if missing:
        raise ContractError(f"missing required fields: {', '.join(missing)}")


def _bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise ContractError(f"{field} must be boolean")
    return value


def _non_negative(value: Any, field: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ContractError(f"{field} must be a non-negative number")
