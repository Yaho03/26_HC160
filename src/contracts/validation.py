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


def check_schema_shape(record: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """
    JSON Schema의 구조 규칙 일부를 의존성 없이 검사한다.

    저장소는 jsonschema를 잠금 의존성에 두지 않는다. 전체 검증이 필요한 곳에서는
    jsonschema가 설치돼 있을 때 함께 돌리고, 이 함수는 어느 환경에서나 도는
    최소 보장을 맡는다.

    검사 범위는 required, additionalProperties, enum, 기본 type, 중첩 object다.
    pattern, minimum, oneOf 등은 검사하지 않으므로 전체 검증을 대체하지 않는다.
    """
    _check_node(record, schema, "")


_JSON_TYPES = {
    "object": dict,
    "array": (list, tuple),
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}


def _type_matches(value: Any, expected) -> bool:
    names = expected if isinstance(expected, list) else [expected]
    for name in names:
        if name == "null":
            if value is None:
                return True
            continue
        if name == "integer" and isinstance(value, bool):
            continue
        if name == "number" and isinstance(value, bool):
            continue
        python_type = _JSON_TYPES.get(name)
        if python_type and isinstance(value, python_type):
            return True
    return False


def _check_node(record: Any, schema: Mapping[str, Any], trail: str) -> None:
    where = trail or "<root>"

    if "enum" in schema and record not in schema["enum"]:
        raise ValueError(f"{where}: {record!r}는 허용값 {schema['enum']}에 없다")

    if "type" in schema and not _type_matches(record, schema["type"]):
        raise ValueError(f"{where}: 타입이 {schema['type']}이어야 한다 (실제 {type(record).__name__})")

    if not isinstance(record, Mapping):
        return

    for field in schema.get("required", []):
        if field not in record:
            raise ValueError(f"{where}: 필수 필드 '{field}'가 없다")

    properties = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        extra = sorted(set(record) - set(properties))
        if extra:
            raise ValueError(f"{where}: 스키마에 없는 필드 {extra}")

    for field, value in record.items():
        subschema = properties.get(field)
        if isinstance(subschema, Mapping):
            _check_node(value, subschema, f"{trail}.{field}" if trail else field)
