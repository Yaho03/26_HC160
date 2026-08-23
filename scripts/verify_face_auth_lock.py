from __future__ import annotations

import re
from pathlib import Path


DIRECT_REQUIREMENTS = Path("requirements-face-auth.txt")
LOCK_FILE = Path("requirements-face-auth.lock")
TARGET_MARKER = "# Target: CPython 3.11 on linux/amd64 (GitHub ubuntu-latest)."
PIN_PATTERN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)")
HASH_PATTERN = re.compile(r"--hash=sha256:([a-f0-9]{64})(?:\s+\\)?$")


class LockValidationError(ValueError):
    pass


def validate_lock(direct_path: Path, lock_path: Path) -> tuple[int, int]:
    direct = _read_pins(direct_path, require_hashes=False)
    locked = _read_pins(lock_path, require_hashes=True)
    lock_text = lock_path.read_text(encoding="utf-8")
    if TARGET_MARKER not in lock_text.splitlines()[:8]:
        raise LockValidationError("lock target must be CPython 3.11 linux/amd64")

    mismatches = []
    for name, version in direct.items():
        if locked.get(name) != version:
            mismatches.append(f"{name}=={version} (locked: {locked.get(name)})")
    if mismatches:
        raise LockValidationError(
            "direct requirements do not match lock: " + ", ".join(mismatches)
        )
    return len(direct), len(locked)


def _read_pins(path: Path, *, require_hashes: bool) -> dict[str, str]:
    pins: dict[str, str] = {}
    hashes: dict[str, int] = {}
    current_name: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        match = PIN_PATTERN.match(line)
        if match:
            name = _normalize_name(match.group(1))
            if name in pins:
                raise LockValidationError(f"duplicate requirement in {path}: {name}")
            pins[name] = match.group(2)
            hashes[name] = 0
            current_name = name
            continue
        if line.startswith("--hash="):
            if current_name is None or HASH_PATTERN.fullmatch(line) is None:
                raise LockValidationError(f"invalid SHA-256 entry in {path}: {line}")
            hashes[current_name] += 1
        elif line and not line.startswith("#") and line != "\\":
            if not line.startswith("--"):
                current_name = None

    if not pins:
        raise LockValidationError(f"no exact requirement pins found in {path}")
    if require_hashes:
        missing = sorted(name for name, count in hashes.items() if count == 0)
        if missing:
            raise LockValidationError(
                "locked requirements missing SHA-256: " + ", ".join(missing)
            )
    return pins


def _normalize_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def main() -> int:
    direct_count, locked_count = validate_lock(DIRECT_REQUIREMENTS, LOCK_FILE)
    print(
        f"face-auth lock valid: {direct_count} direct, "
        f"{locked_count} total packages"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
