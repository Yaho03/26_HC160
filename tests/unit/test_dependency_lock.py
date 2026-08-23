import tempfile
import unittest
from pathlib import Path

from scripts.verify_face_auth_lock import (
    LockValidationError,
    TARGET_MARKER,
    validate_lock,
)


class DependencyLockTest(unittest.TestCase):
    def test_repository_lock_matches_direct_requirements(self):
        direct_count, locked_count = validate_lock(
            Path("requirements-face-auth.txt"), Path("requirements-face-auth.lock")
        )

        self.assertEqual(direct_count, 8)
        self.assertGreater(locked_count, direct_count)

    def test_direct_version_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct = root / "requirements.txt"
            lock = root / "requirements.lock"
            direct.write_text("example==2.0\n", encoding="utf-8")
            lock.write_text(
                f"{TARGET_MARKER}\n"
                "example==1.0 \\\n"
                f"    --hash=sha256:{'a' * 64}\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(LockValidationError, "do not match"):
                validate_lock(direct, lock)

    def test_missing_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct = root / "requirements.txt"
            lock = root / "requirements.lock"
            direct.write_text("example==1.0\n", encoding="utf-8")
            lock.write_text(
                f"{TARGET_MARKER}\nexample==1.0\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(LockValidationError, "missing SHA-256"):
                validate_lock(direct, lock)

    def test_wrong_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct = root / "requirements.txt"
            lock = root / "requirements.lock"
            direct.write_text("example==1.0\n", encoding="utf-8")
            lock.write_text(
                "# Target: CPython 3.11 on linux/arm64.\n"
                "example==1.0 \\\n"
                f"    --hash=sha256:{'a' * 64}\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(LockValidationError, "linux/amd64"):
                validate_lock(direct, lock)


if __name__ == "__main__":
    unittest.main()
